"""P5b measurement harness: strata-minimax vs llama.cpp on the same GGUF.

Both engines are measured on the same file, the same prompt lengths and the same box, and the box load (free RAM,
per-GPU VRAM, per-GPU utilisation) is sampled immediately before each run so a number can be read against the state
it was taken in.

    python tools/minimax/bench_p5b.py --model m.gguf --lengths 512,4096,32768 \
        --llama-bench R:/build/bin/llama-bench.exe --llama-dev CUDA1 \
        -- --gpus 1 --minimax-kv vram --minimax-prefill-batch 4096

Everything after `--` goes to strata-minimax. `--skip-engine` / `--skip-llama` run one side only.

The engine numbers are its own PP / DONE lines (tools/minimax/bench_p5a.py reads the same ones). The llama.cpp
numbers come from `llama-bench -o json`: `pp<N>` is prefill, `tg<N>` is decode at that prompt length.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def exe() -> str:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    raise SystemExit('strata-minimax is not built')


def ids(n: int, vocab: int, seed: int) -> list[int]:
    x = seed
    out = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x % vocab)
    return out


def load_line() -> str:
    """Free RAM, per-GPU VRAM and utilisation right now."""
    ram = 'ram=?'
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram = 'ram_free=%.1fGiB' % (m.ullAvailPhys / (1 << 30))
    except Exception:
        pass
    gpu = ''
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
        gpu = ' gpu=' + ';'.join(x.strip().replace(', ', '/') for x in out.stdout.splitlines())
    except Exception:
        pass
    return ram + gpu


def run_engine(model: str, prompt: list[int], max_new: int, extra: list[str], ctx: int, timeout: int):
    cmd = [exe(), '--model', model, '--serve', '--max-context', str(ctx), *extra]
    text = f'GEN {max_new} temperature=0 ' + ' '.join(map(str, prompt)) + '\nSTOP\n'
    p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=timeout)
    if p.returncode != 0:
        raise SystemExit(p.stderr[-3000:])
    done = [l for l in p.stdout.splitlines() if l.startswith('DONE ')]
    if not done:
        raise SystemExit('no DONE line:\n' + p.stdout[-1000:] + p.stderr[-3000:])
    f = done[0].split()
    gen, pr, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
    finite = f[7] if len(f) > 7 else '?'
    return gen, pr, pms, dms, finite, p.stderr


def run_llama(bench: str, model: str, n_prompt: int, n_gen: int, reps: int, threads: int, dev: str,
              extra: list[str], timeout: int) -> dict:
    cmd = [bench, '-m', model, '-p', str(n_prompt), '-n', str(n_gen), '-r', str(reps), '-t', str(threads),
           '-ngl', '99', '-fa', '1', '-o', 'json', *extra]
    if dev:
        cmd += ['-dev', dev]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    if p.returncode != 0:
        raise SystemExit('llama-bench failed:\n' + p.stderr[-3000:] + p.stdout[-2000:])
    txt = p.stdout[p.stdout.find('['):]
    rows = json.loads(txt)
    out = {}
    for r in rows:
        key = r.get('test')
        if key is None:
            key = ('pp%d' % r['n_prompt']) if r.get('n_prompt') else 'tg'
        out[key] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--id-range', type=int, default=200000)
    ap.add_argument('--lengths', default='512,4096')
    ap.add_argument('--decode', type=int, default=32)
    ap.add_argument('--ctx', type=int, default=32768)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--llama-bench')
    ap.add_argument('--llama-dev', default='CUDA1')
    ap.add_argument('--llama-args', default='-ot exps=CPU')
    ap.add_argument('--llama-threads', type=int, default=16)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=14400)
    ap.add_argument('--skip-engine', action='store_true')
    ap.add_argument('--skip-llama', action='store_true')
    ap.add_argument('--json-out')
    ap.add_argument('engine_args', nargs='*')
    a = ap.parse_args()
    extra = [x for x in a.engine_args if x != '--']
    llama_extra = a.llama_args.split()
    rows = []
    print(f'engine argv: {" ".join(extra)}')
    print(f'llama argv:  {" ".join(llama_extra)} dev={a.llama_dev} t={a.llama_threads}')
    for n in [int(x) for x in a.lengths.split(',')]:
        prompt = ids(n, a.id_range, a.seed)
        row = {'prompt': n}
        if not a.skip_engine:
            print(f'[{n}] load before engine: {load_line()}', flush=True)
            gen, pr, pms, dms, finite, err = run_engine(a.model, prompt, a.decode, extra, a.ctx, a.timeout)
            row.update(engine_prefill=pr / (pms / 1000.0), engine_ttft_ms=pms,
                       engine_decode=gen / (dms / 1000.0), engine_ms_per_token=dms / max(gen, 1),
                       engine_gen=gen, engine_finish=finite)
            print(f'[{n}] engine prefill={row["engine_prefill"]:.1f} tok/s ttft={pms/1000.0:.2f}s '
                  f'decode={row["engine_decode"]:.2f} tok/s ({row["engine_ms_per_token"]:.1f} ms/tok) finish={finite}',
                  flush=True)
        if not a.skip_llama and a.llama_bench:
            print(f'[{n}] load before llama.cpp: {load_line()}', flush=True)
            res = run_llama(a.llama_bench, a.model, n, a.decode, a.reps, a.llama_threads, a.llama_dev, llama_extra,
                            a.timeout)
            pp, tg = res.get(f'pp{n}'), res.get('tg')
            if pp:
                row.update(llama_prefill=pp['avg_ts'], llama_ttft_ms=n / pp['avg_ts'] * 1000.0)
            if tg:
                row.update(llama_decode=tg['avg_ts'], llama_ms_per_token=1000.0 / tg['avg_ts'])
            print(f'[{n}] llama.cpp prefill={row.get("llama_prefill", float("nan")):.1f} tok/s '
                  f'ttft={row.get("llama_ttft_ms", float("nan"))/1000.0:.2f}s '
                  f'decode={row.get("llama_decode", float("nan")):.2f} tok/s '
                  f'({row.get("llama_ms_per_token", float("nan")):.1f} ms/tok)', flush=True)
        rows.append(row)
    print()
    head = f'{"prompt":>7} | {"engine pp":>9} {"llama pp":>8} | {"engine tg":>9} {"llama tg":>8} | {"engine TTFT":>11} {"llama TTFT":>10}'
    print(head)
    print('-' * len(head))
    for r in rows:
        def g(k, f='%.1f'):
            v = r.get(k)
            return f % v if isinstance(v, (int, float)) else '—'
        print(f'{r["prompt"]:7d} | {g("engine_prefill"):>9} {g("llama_prefill"):>8} | '
              f'{g("engine_decode", "%.2f"):>9} {g("llama_decode", "%.2f"):>8} | '
              f'{g("engine_ttft_ms", "%.0f"):>11} {g("llama_ttft_ms", "%.0f"):>10}')
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({'slots': rows, 'engine_args': extra, 'llama_args': llama_extra,
                                                'llama_dev': a.llama_dev}, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
