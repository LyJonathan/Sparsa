"""P5c full-model measurement harness: strata-minimax vs llama.cpp on the whole 146 GiB checkpoint.

What this adds over bench_p5b.py:
  * one engine process per context, several GEN requests inside it, so the pinned expert arena is filled once
    and the repeated requests measure the deployed (warm) state instead of paying a 90 GiB refill each time.
  * every rep gets a different last prompt token, so the engine's own prefix-cache reuse (`DONE ... reused`)
    cannot turn a repeat into a cache hit; each rep prefills from a reset cache.
  * peak VRAM per GPU, peak process working set and the minimum free system RAM are sampled while the run
    is in flight, so the memory column is a measurement, not an estimate.
  * the tier counters the engine prints (`--minimax-stats total ...`) are carried into the JSON.

    python tools/minimax/bench_p5c.py --model m.gguf --ids prompt_ids.txt \
        --lengths 512,4096 --reps 3 --llama-bench R:/build/bin/llama-bench.exe \
        -- --gpus 0,1 --layer-split auto --minimax-kv vram --minimax-dense-layers-gpu -1

Ids: `--ids FILE` is a comma-separated token-id list, cycled up to each length (`mk_prompt.py` builds one from
the profile-building prompts). Without `--ids`, synthetic ids come from the same LCG bench_p5b.py uses, which is
what llama-bench feeds itself - so the two engines see the same token distribution.

Rep 0 of every context is the first request to a freshly started engine (cold: the arena fill and the page cache
are as the box left them); reps 1+ are warm.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

GPU_COLUMNS = 'index,memory.used,pcie.link.gen.current,pcie.link.width.current'


def exe() -> str:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    raise SystemExit('strata-minimax is not built')


def synth_ids(n: int, vocab: int, seed: int) -> list[int]:
    """The same LCG bench_p5b.py uses; llama-bench draws its own random ids over the vocabulary."""
    x = seed
    out = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x % vocab)
    return out


def prompt_for(source: list[int], n: int, rep: int) -> list[int]:
    """n ids from `source`, cycled, with the last one moved so a repeat cannot hit the prefix cache."""
    if not source:
        raise SystemExit('empty id list')
    out = [source[i % len(source)] for i in range(n)]
    out[-1] = (out[-1] + rep + 1) % 200000
    return out


class ProcessMem(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_ram_gib() -> float:
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / (1 << 30)


def proc_working_set_mib(pid: int) -> float:
    p = ProcessMem()
    p.cb = ctypes.sizeof(ProcessMem)
    h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0010, False, pid)   # QUERY_LIMITED_INFORMATION
    if not h:
        return 0.0
    try:
        if not ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(p), p.cb):
            return 0.0
        return p.WorkingSetSize / (1 << 20)
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def gpu_state() -> str:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=' + GPU_COLUMNS,
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
        return ';'.join(x.strip() for x in out.stdout.splitlines())
    except Exception:
        return '?'


class Sampler(threading.Thread):
    """Polls VRAM per GPU, the process working set and free RAM while a run is in flight."""

    def __init__(self, period: float = 0.5):
        super().__init__(daemon=True)
        self.period = period
        self.pid = 0
        self.stop = threading.Event()
        self.peak_gpu: dict[int, float] = {}
        self.peak_ws_mib = 0.0
        self.min_free_ram_gib = float('inf')

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=30)
                for line in out.stdout.splitlines():
                    if not line.strip():
                        continue
                    i, mb = [x.strip() for x in line.split(',')]
                    k, v = int(i), float(mb)
                    self.peak_gpu[k] = max(self.peak_gpu.get(k, 0.0), v)
            except Exception:
                pass
            if self.pid:
                self.peak_ws_mib = max(self.peak_ws_mib, proc_working_set_mib(self.pid))
            r = free_ram_gib()
            if r > 0:
                self.min_free_ram_gib = min(self.min_free_ram_gib, r)
            self.stop.wait(self.period)

    def result(self, devs: list[int]) -> dict:
        d = {f'peak_gpu{g}_mib': round(self.peak_gpu.get(g, 0.0)) for g in devs}
        d['peak_ws_mib'] = round(self.peak_ws_mib)
        if self.min_free_ram_gib != float('inf'):
            d['min_free_ram_gib'] = round(self.min_free_ram_gib, 1)
        return d


def parse_stats(line: str) -> dict | None:
    """INFO minimax-stats total token=32 vram=10 ram=80 ssd=138 ssd_bytes=... swaps=0"""
    if 'minimax-stats' not in line:
        return None
    out = {'total': 'total' in line}
    for tok in line.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            try:
                out[k] = int(v)
            except ValueError:
                pass
    return out if len(out) > 1 else None


def run_engine(model: str, ctx: int, reps: int, decode: int, source: list[int], n: int,
               extra: list[str], timeout: int, devs: list[int]) -> list[dict]:
    """One fresh engine process for one context, then `reps` requests: rep 0 is the cold one."""
    cmd = [exe(), '--model', model, '--serve', '--max-context', str(ctx), *extra]
    print(f'engine argv: {" ".join(cmd)}', flush=True)
    samp = Sampler()
    samp.start()
    t_launch = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding='utf-8', errors='replace', bufsize=1)
    samp.pid = p.pid
    stats: list[dict] = []
    err: list[str] = []

    def reader():
        shown = 0
        for line in p.stderr:
            err.append(line.rstrip())
            s = parse_stats(line)
            if s:
                stats.append(s)
            elif shown < 64:
                shown += 1
                print('  engine: ' + line.rstrip(), flush=True)

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    out: list[dict] = []
    lines = []
    for rep in range(reps):
        ids = prompt_for(source, n, rep)
        lines.append(f'GEN {decode} temperature=0 ' + ' '.join(map(str, ids)) + '\n')
    lines.append('STOP\n')
    try:
        p.stdin.write(''.join(lines))
        p.stdin.flush()
        p.stdin.close()     # EOF: the engine exits once the queue is drained
    except BrokenPipeError:
        p.wait(timeout=60)
        samp.stop.set()
        raise SystemExit('engine closed stdin (exited rc=%s) before the first request:\n%s'
                         % (p.returncode, '\n'.join(err[-40:])))

    done = []
    ready_s = 0.0
    for line in p.stdout:
        if line.startswith('READY'):
            ready_s = time.time() - t_launch
            print(f'engine READY after {ready_s:.0f}s', flush=True)
        if line.startswith('DONE '):
            done.append(line.split())
    p.stdin.close()
    rc = p.wait(timeout=timeout)
    samp.stop.set()
    samp.join(timeout=5)
    print(f'engine exited rc={rc} after {time.time() - t_launch:.0f}s', flush=True)
    if rc != 0 and not done:
        raise SystemExit('engine failed:\n' + '\n'.join(err[-40:]))

    st = [s for s in stats if s.get('total') and s.get('token', 0) > 0]
    last_total = st[-1] if st else {}
    peak = samp.result(devs)
    for rep, f in enumerate(done):
        gen, pr, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
        reused = int(f[8]) if len(f) > 8 else -1
        row = {'rep': rep, 'engine_prefill': pr / (pms / 1000.0) if pms > 0 else 0.0,
               'engine_ttft_ms': pms, 'engine_decode': gen / (dms / 1000.0) if dms > 0 else 0.0,
               'engine_ms_per_token': dms / gen if gen else 0.0, 'engine_gen': gen,
               'engine_finish': f[5] if len(f) > 5 else '?', 'engine_reused': reused}
        if rep == 0 and last_total:
            for k in ('vram', 'ram', 'ssd', 'ssd_bytes', 'swaps', 'token'):
                if k in last_total:
                    row['tier_' + k] = last_total[k]
        row.update(peak)
        row['engine_ready_s'] = round(ready_s, 1)
        out.append(row)
        print(f'[{n}] rep{rep} prefill={row["engine_prefill"]:.1f} tok/s ttft={pms/1000.0:.2f}s '
              f'decode={row["engine_decode"]:.2f} tok/s ({row["engine_ms_per_token"]:.1f} ms/tok) '
              f'finish={row["engine_finish"]} reused={reused} '
              f'tiers vram={row.get("tier_vram", "?")} ram={row.get("tier_ram", "?")} '
              f'ssd={row.get("tier_ssd", "?")} ssd_GiB={row.get("tier_ssd_bytes", 0)/2**30:.1f}', flush=True)
    Path('bench_p5c_engine.err').write_text('\n'.join(err), encoding='utf-8')
    return out


def run_llama(bench: str, model: str, n_prompt: int, n_gen: int, reps: int, threads: int, dev: str,
              extra: list[str], timeout: int, devs: list[int]) -> dict:
    cmd = [bench, '-m', model, '-p', str(n_prompt), '-n', str(n_gen), '-r', str(reps), '-t', str(threads),
           '-ngl', '99', '-fa', 'on', '-sm', 'layer', '-o', 'json', *extra]
    if dev:
        cmd += ['-dev', dev]
    print(f'llama argv: {" ".join(cmd)}', flush=True)
    samp = Sampler()
    samp.start()
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding='utf-8', errors='replace')
    samp.pid = p.pid
    so, se = p.communicate(timeout=timeout)
    samp.stop.set()
    samp.join(timeout=5)
    print(f'llama exited rc={p.returncode} after {time.time() - t0:.0f}s', flush=True)
    if p.returncode != 0:
        raise SystemExit('llama-bench failed:\n' + se[-3000:] + so[-2000:])
    rows = json.loads(so[so.find('['):])
    out = {}
    for r in rows:
        key = r.get('test') or ('pp%d' % r['n_prompt'] if r.get('n_prompt') else 'tg')
        out[key] = r
    res = {}
    pp, tg = out.get(f'pp{n_prompt}'), out.get('tg')
    if pp:
        res.update(llama_prefill=pp['avg_ts'], llama_prefill_min=pp['min_ts'], llama_prefill_max=pp['max_ts'],
                   llama_ttft_ms=n_prompt / pp['avg_ts'] * 1000.0)
    if tg:
        res.update(llama_decode=tg['avg_ts'], llama_decode_min=tg['min_ts'], llama_decode_max=tg['max_ts'],
                   llama_ms_per_token=1000.0 / tg['avg_ts'])
    res.update(samp.result(devs))
    print(f'[{n_prompt}] llama prefill={res.get("llama_prefill", float("nan")):.1f} tok/s '
          f'decode={res.get("llama_decode", float("nan")):.2f} tok/s', flush=True)
    return res


def http_json(url: str, payload=None, timeout: float = 60.0):
    import urllib.request
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def run_llama_server(binary: str, model: str, ctx: int, reps: int, decode: int, lengths: list[int],
                     prompts_dir: str, extra: list[str], threads: int, timeout: int, devs: list[int],
                     port: int) -> dict[int, list[dict]]:
    """llama-server with the same text prompts: one model load, one request per rep (cache_prompt off)."""
    cmd = [binary, '-m', model, '-c', str(ctx), '-ngl', '99', '-fa', 'on', '-sm', 'layer', '-t', str(threads),
           '--host', '127.0.0.1', '--port', str(port), '--parallel', '1', *extra]
    print(f'llama-server argv: {" ".join(cmd)}', flush=True)
    log = open('bench_p5c_llama.log', 'w', encoding='utf-8')
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    base = f'http://127.0.0.1:{port}'
    up = False
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            raise SystemExit('llama-server exited during load; see bench_p5c_llama.log')
        try:
            http_json(base + '/health', timeout=5)
            up = True
            break
        except Exception:
            time.sleep(5)
    if not up:
        p.terminate()
        raise SystemExit('llama-server did not come up; see bench_p5c_llama.log')
    print(f'llama-server up after {time.time() - t0:.0f}s', flush=True)

    out: dict[int, list[dict]] = {}
    try:
        for n in lengths:
            text = (Path(prompts_dir) / f'text_{n}.txt').read_text(encoding='utf-8')
            rows = []
            for rep in range(reps):
                samp = Sampler()
                samp.start()
                samp.pid = p.pid
                t1 = time.time()
                r = http_json(base + '/completion', {'prompt': text, 'n_predict': decode, 'temperature': 0,
                                                    'cache_prompt': False, 'stream': False}, timeout=timeout)
                wall = time.time() - t1
                samp.stop.set()
                samp.join(timeout=5)
                tm = r.get('timings') or {}
                row = {'rep': rep,
                       'llama_prefill': tm.get('prompt_per_second', 0.0),
                       'llama_ttft_ms': tm.get('prompt_ms', 0.0),
                       'llama_decode': tm.get('predicted_per_second', 0.0),
                       'llama_ms_per_token': 1000.0 / tm['predicted_per_second'] if tm.get('predicted_per_second') else 0.0,
                       'llama_prompt_n': tm.get('prompt_n'), 'llama_gen_n': tm.get('predicted_n'),
                       'llama_wall_s': round(wall, 1)}
                row.update(samp.result(devs))
                rows.append(row)
                print(f'[{n}] llama rep{rep} prefill={row["llama_prefill"]:.1f} tok/s '
                      f'ttft={row["llama_ttft_ms"]/1000.0:.2f}s decode={row["llama_decode"]:.2f} tok/s '
                      f'({row["llama_ms_per_token"]:.1f} ms/tok) prompt_n={row["llama_prompt_n"]} '
                      f'gen={row["llama_gen_n"]} wall={row["llama_wall_s"]}s', flush=True)
            out[n] = rows
    finally:
        p.terminate()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ids')
    ap.add_argument('--vocab', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--lengths', default='512,4096')
    ap.add_argument('--ctx', type=int, default=32768)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--decode', type=int, default=32)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--llama-bench')
    ap.add_argument('--llama-server')
    ap.add_argument('--prompts-dir', default='bench-prompts')
    ap.add_argument('--port', type=int, default=8099)
    ap.add_argument('--llama-dev', default='CUDA0/CUDA1')
    ap.add_argument('--llama-args', default='-ot exps=CPU')
    ap.add_argument('--llama-threads', type=int, default=24)
    ap.add_argument('--llama-reps', type=int, default=0)
    ap.add_argument('--timeout', type=int, default=14400)
    ap.add_argument('--skip-engine', action='store_true')
    ap.add_argument('--skip-llama', action='store_true')
    ap.add_argument('--json-out')
    ap.add_argument('engine_args', nargs='*')
    a = ap.parse_args()
    extra = [x for x in a.engine_args if x != '--']
    devs = [int(x) for x in a.gpus.split(',') if x.strip() != '']
    lengths = [int(x) for x in a.lengths.split(',')]

    if a.ids:
        raw = Path(a.ids).read_bytes()
        if a.ids.lower().endswith('.bin'):
            import array
            arr = array.array('i')
            arr.frombytes(raw[:len(raw) // 4 * 4])
            source = list(arr)          # the G3 dump format: int32 little-endian
        else:
            source = [int(x) for x in raw.decode().split(',') if x.strip()]
        print(f'ids: {len(source)} from {a.ids}', flush=True)
    else:
        source = synth_ids(200000, a.vocab, a.seed)

    all_rows: dict[int, dict] = {}
    if not a.skip_engine:
        for n in lengths:
            print(f'[{n}] load before engine: ram_free={free_ram_gib():.1f}GiB gpu={gpu_state()}', flush=True)
            rows = run_engine(a.model, n + a.decode + 1, a.reps, a.decode, source, n, extra, a.timeout, devs)
            all_rows.setdefault(n, {})['prompt'] = n
            all_rows[n]['engine'] = rows
    if not a.skip_llama and a.llama_bench:
        for n in lengths:
            print(f'load before llama.cpp: ram_free={free_ram_gib():.1f}GiB gpu={gpu_state()}', flush=True)
            r = run_llama(a.llama_bench, a.model, n, a.decode, a.llama_reps or a.reps, a.llama_threads,
                          a.llama_dev, a.llama_args.split(), a.timeout, devs)
            all_rows.setdefault(n, {})['prompt'] = n
            all_rows[n]['llama'] = r
    if not a.skip_llama and a.llama_server:
        print(f'load before llama-server: ram_free={free_ram_gib():.1f}GiB gpu={gpu_state()}', flush=True)
        srv = run_llama_server(a.llama_server, a.model, max(lengths) + a.decode + 1, a.llama_reps or a.reps,
                               a.decode, lengths, a.prompts_dir, a.llama_args.split(), a.llama_threads,
                               a.timeout, devs, a.port)
        for n, rows in srv.items():
            all_rows.setdefault(n, {})['prompt'] = n
            all_rows[n]['llama_server'] = rows

    rows = [all_rows[k] for k in sorted(all_rows)]
    print()
    head = (f'{"ctx":>6} | {"arm":>5} | {"prefill":>9} | {"decode":>8} | {"ms/tok":>7} | {"TTFT s":>7} | '
            f'{"GPU0 MiB":>8} {"GPU1 MiB":>8} | {"RAM peak MiB":>12}')
    print(head)
    print('-' * len(head))
    for r in rows:
        for i, e in enumerate(r.get('engine', [])):
            print(f'{r["prompt"]:6d} | {"cold" if i == 0 else "warm":>5} | {e["engine_prefill"]:9.1f} | '
                  f'{e["engine_decode"]:8.2f} | {e["engine_ms_per_token"]:7.1f} | {e["engine_ttft_ms"]/1000:7.2f} | '
                  f'{e.get("peak_gpu0_mib", 0):8.0f} {e.get("peak_gpu1_mib", 0):8.0f} | '
                  f'{e.get("peak_ws_mib", 0):12.0f}')
        l = r.get('llama')
        if l:
            print(f'{r["prompt"]:6d} | {"llama":>5} | {l["llama_prefill"]:9.1f} | {l["llama_decode"]:8.2f} | '
                  f'{l["llama_ms_per_token"]:7.1f} | {l["llama_ttft_ms"]/1000:7.2f} | '
                  f'{l.get("peak_gpu0_mib", 0):8.0f} {l.get("peak_gpu1_mib", 0):8.0f} | '
                  f'{l.get("peak_ws_mib", 0):12.0f}')
        for i, e in enumerate(r.get('llama_server', [])):
            print(f'{r["prompt"]:6d} | {"lcold" if i == 0 else "lwarm":>5} | {e["llama_prefill"]:9.1f} | '
                  f'{e["llama_decode"]:8.2f} | {e["llama_ms_per_token"]:7.1f} | {e["llama_ttft_ms"]/1000:7.2f} | '
                  f'{e.get("peak_gpu0_mib", 0):8.0f} {e.get("peak_gpu1_mib", 0):8.0f} | '
                  f'{e.get("peak_ws_mib", 0):12.0f}')
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({'rows': rows, 'engine_args': extra,
                                                'llama_args': a.llama_args.split(),
                                                'llama_dev': a.llama_dev, 'ids': a.ids or 'synthetic',
                                                'reps': a.reps, 'decode': a.decode}, indent=1), encoding='utf-8')
        print(f'wrote {a.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
