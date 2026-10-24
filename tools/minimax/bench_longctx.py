"""Long-context measurement: a real-text prompt through the batched prompt pass, then a short decode.

Round 1 could not finish 32 K, and the run that came closest left the desktop with 0.3 GB of RAM.  This harness
exists so that run is reproducible and is measured the same way every time:

* the prompt is **real text** tokenized with the model's own `tokenizer.json` (not random ids), truncated or
  repeated to exactly the requested length;
* the engine's own `PP` / `DONE` lines give prefill and decode (no wall-clock guessing);
* `\\Memory\\Available MBytes` and the engine's working set are sampled while it runs, so "did it starve the
  desktop" is a number, not an impression;
* the whole run is killed at `--cap-s`, so a run that will not finish reports its per-chunk timings and stops.

    python tools/minimax/bench_longctx.py --model m.gguf --text docs/DETAILS.md --length 32768 \
        --decode 64 --cap-s 1500 -- --gpus 1 --minimax-tiers --minimax-kv ram --minimax-prefill-batch 32768

Everything after `--` goes to the engine.  At the end one JSON object is printed and appended to --jsonl.
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


def exe() -> str:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build-lc2' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax.exe', ROOT / 'build-lc2' / 'strata-minimax',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    raise SystemExit('strata-minimax is not built')


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def available_mib() -> float:
    if os.name == 'nt':
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 1048576.0
    with open('/proc/meminfo', encoding='utf-8') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024.0
    return float('nan')


def prompt_ids(text_path: str, n: int, tok_path: str) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    text = Path(text_path).read_text(encoding='utf-8', errors='replace')
    ids = tok.encode(text).ids
    if not ids:
        raise SystemExit(f'{text_path} tokenized to nothing')
    while len(ids) < n:
        ids = ids + ids           # a repeated real document, not random ids
    return ids[:n]


def run(model: str, prompt: list[int], max_new: int, extra: list[str], ctx: int, cap_s: float):
    cmd = [exe(), '--model', model, '--serve', '--max-context', str(ctx), *extra]
    feed = f'GEN {max_new} temperature=0 ' + ' '.join(map(str, prompt)) + '\nSTOP\n'
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding='utf-8', errors='replace', bufsize=1)
    lines: list[str] = []
    err: list[str] = []
    stop = threading.Event()

    def rd_out():
        for line in p.stdout:
            lines.append(line.rstrip('\n'))
        stop.set()

    def rd_err():
        for line in p.stderr:
            err.append(line.rstrip('\n'))

    threading.Thread(target=rd_out, daemon=True).start()
    threading.Thread(target=rd_err, daemon=True).start()
    try:
        p.stdin.write(feed)
        p.stdin.flush()
        p.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    import psutil
    proc = psutil.Process(p.pid)
    peak_ws = 0.0
    min_avail = float('inf')
    samples: list[tuple[float, float, float]] = []
    killed = False
    while not stop.is_set():
        try:
            ws = proc.memory_info().rss / 1073741824.0
        except psutil.Error:
            ws = 0.0
        av = available_mib() / 1024.0
        peak_ws = max(peak_ws, ws)
        min_avail = min(min_avail, av)
        samples.append((time.time() - t0, ws, av))
        if time.time() - t0 > cap_s:
            killed = True
            p.kill()
            break
        stop.wait(2.0)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    wall = time.time() - t0

    done = [l for l in lines if l.startswith('DONE ')]
    pp = [l for l in lines if l.startswith('PP ')]
    out = {'wall_s': round(wall, 1), 'peak_working_set_gib': round(peak_ws, 2),
           'min_available_gib': round(min_avail, 2), 'killed_at_cap': killed,
           'pp_lines': len(pp), 'exit': p.returncode}
    if done:
        f = done[0].split()
        gen, pr, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
        out.update({'generated': gen, 'prompt_tokens': pr, 'prefill_ms': pms, 'decode_ms': dms,
                    'prefill_tps': round(pr / (pms / 1000.0), 2) if pms > 0 else None,
                    'decode_tps': round(gen / (dms / 1000.0), 3) if dms > 0 else None})
    else:
        out['last_pp'] = pp[-1] if pp else None
    # The engine's own plan lines say what the batch, workspace and read path actually were.
    plan = [l for l in err if 'minimax-prefill' in l or 'minimax-ram-tier' in l or 'minimax-vram-slots' in l
            or 'minimax-prefill-workspace' in l or 'minimax-kv' in l]
    out['plan'] = plan[:20]
    out['stderr_tail'] = err[-15:]
    out['stdout_tail'] = lines[-5:]
    return out, samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True, help='a real document to tokenize into the prompt')
    ap.add_argument('--tokenizer',
                    default="E:/Jon's Projects/AI/Strata Fork Minimax/models/MiniMax-M3.1-Flash/tokenizer.json")
    ap.add_argument('--length', type=int, default=32768)
    ap.add_argument('--decode', type=int, default=64)
    ap.add_argument('--ctx', type=int, default=0, help='0: the prompt length')
    ap.add_argument('--cap-s', type=float, default=1500.0)
    ap.add_argument('--jsonl', default='')
    ap.add_argument('--dump-ids', default='', help='write the token ids here for a repeat run')
    ap.add_argument('--ids-file', default='', help='use these token ids instead of tokenizing --text')
    ap.add_argument('engine_args', nargs='*')
    a = ap.parse_args()
    extra = [x for x in a.engine_args if x != '--']
    if a.ids_file:
        ids = [int(x) for x in
               Path(a.ids_file).read_text(encoding='utf-8-sig').replace('\n', ' ').split()]
    else:
        ids = prompt_ids(a.text, a.length, a.tokenizer)
    if a.dump_ids:
        Path(a.dump_ids).write_text(' '.join(map(str, ids)), encoding='utf-8')
    ctx = a.ctx if a.ctx > 0 else a.length
    print(f'engine: {exe()}', file=sys.stderr)
    print(f'argv: {" ".join(extra)}', file=sys.stderr)
    out, samples = run(a.model, ids, a.decode, extra, ctx, a.cap_s)
    out['length'] = a.length
    out['argv'] = extra
    if samples:
        out['samples'] = [[round(t, 1), round(w, 2), round(v, 2)] for t, w, v in samples[::max(1, len(samples) // 40)]]
    print(json.dumps(out, indent=1))
    if a.jsonl:
        with open(a.jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(out) + '\n')
    return 0 if ('prefill_ms' in out and not out['killed_at_cap']) else 1


if __name__ == '__main__':
    sys.exit(main())
