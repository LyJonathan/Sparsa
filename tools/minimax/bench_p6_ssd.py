"""P6 SSD-path A/B: run one engine binary over a model whose experts do not fit, and report the decode rate
together with the tier counters (including the time a token waited for the drive).

The point of the harness is the SSD tier: `--minimax-ram-experts-per-layer` is set small so most routes miss
both the VRAM and the RAM tier and are read from the file, which is what a model larger than RAM does to every
token.  Two binaries are compared by running them one after the other, interleaved, in the same process-free
state.

When an arm runs with `--minimax-profile-time`, the per-token phase lines (`INFO minimax-profile ...`) are
collected too and printed as a phase table after the decode table: that is the P8 breakdown (attention, router,
the CPU lane's passes and their achieved GB/s, the drive wait, the combine, the shared expert, the RAM-tier
admission, lm_head, and `other`).

    python tools/minimax/bench_p6_ssd.py --exe A/build/strata-minimax.exe --exe B/build/strata-minimax.exe \
        --model slice.gguf --prompt 512 --decode 32 --reps 3 -- --gpus 1 --minimax-tiers \
        --minimax-vram-slots-per-layer 0 --minimax-ram-experts-per-layer 4

Every arm gets a fresh engine process (the arena and the page cache are as the box left them), and each rep's
last prompt token differs so the engine's prefix cache cannot turn a rep into a hit.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

VOCAB = 200000


def synth_ids(n: int, seed: int) -> list[int]:
    x = seed
    out = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x % VOCAB)
    return out


def prompt_for(source: list[int], n: int, rep: int) -> list[int]:
    out = [source[i % len(source)] for i in range(n)]
    out[-1] = (out[-1] + rep + 1) % VOCAB
    return out


def read_ids(path: str) -> list[int]:
    """int32 (the G3 dump format) or whitespace-separated text."""
    raw = Path(path).read_bytes()
    if path.lower().endswith('.bin') and len(raw) % 4 == 0:
        import numpy as np
        return [int(x) for x in np.frombuffer(raw, dtype=np.int32)]
    return [int(x) for x in raw.decode('utf-8', 'replace').replace(',', ' ').split() if x.lstrip('-').isdigit()]


NUM = ('vram', 'ram', 'ssd', 'ssd_bytes', 'skipped', 'swaps', 'pcie', 'token')


def parse_line(line: str) -> dict | None:
    if 'minimax-stats' not in line:
        return None
    d = {'total': 'total' in line}
    for tok in line.split():
        if '=' not in tok:
            continue
        k, v = tok.split('=', 1)
        try:
            d[k] = int(v)
        except ValueError:
            try:
                d[k] = float(v)
            except ValueError:
                pass
    return d if len(d) > 1 else None


def parse_profile(line: str) -> dict | None:
    """A `--minimax-profile-time` line: phase=ms pairs, token=N.  The `--minimax-stats` totals line repeats the mean
    of those tokens with the same key names, so it is excluded here (it is parsed as a stats row)."""
    if 'minimax-profile' not in line or 'minimax-stats' in line:
        return None
    d = {}
    for tok in line.split():
        if '=' not in tok:
            continue
        k, v = tok.split('=', 1)
        try:
            d[k] = float(v)
        except ValueError:
            pass
    return d if 'total' in d else None


PHASES = ('total', 'embed', 'attn', 'idx_sync', 'msa_sync', 'router', 'pick', 'gpu_begin', 'cpu_ram', 'ssd_wait',
          'cpu_ssd', 'gpu_finish', 'gpu_stream', 'combine', 'shared', 'admit', 'prefetch', 'lm_head', 'other')


def run_arm(exe: str, model: str, ctx: int, prompt: int, decode: int, reps: int, source: list[int],
            extra: list[str], timeout: int) -> dict:
    cmd = [exe, '--model', model, '--serve', '--max-context', str(ctx), *extra]
    print(f'  argv: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding='utf-8', errors='replace', bufsize=1)
    stats: list[dict] = []
    waits: list[float] = []
    profiles: list[dict] = []
    err: list[str] = []

    def reader():
        for line in p.stderr:
            err.append(line.rstrip())
            prof = parse_profile(line)
            if prof is not None:
                profiles.append(prof)
                continue
            s = parse_line(line)
            if s is None:
                continue
            stats.append(s)
            if 'ssd_wait_ms' in s and 'token' in s:
                waits.append(float(s['ssd_wait_ms']))

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    lines = [f'GEN {decode} temperature=0 ' + ' '.join(map(str, prompt_for(source, prompt, r))) + '\n'
             for r in range(reps)] + ['STOP\n']
    try:
        p.stdin.write(''.join(lines))
        p.stdin.flush()
        p.stdin.close()
    except BrokenPipeError:
        p.wait(timeout=120)
        raise SystemExit('engine closed stdin before the first request:\n' + '\n'.join(err[-30:]))

    done = []
    for line in p.stdout:
        if line.startswith('DONE '):
            done.append(line.split())
    rc = p.wait(timeout=timeout)
    if rc != 0 and not done:
        raise SystemExit(f'engine failed rc={rc}:\n' + '\n'.join(err[-40:]))

    rows = []
    for i, f in enumerate(done):
        gen, pms, dms = int(f[1]), float(f[3]), float(f[4])
        rows.append({'rep': i, 'decode': gen / (dms / 1000.0) if dms else 0.0,
                     'ms_per_token': dms / gen if gen else 0.0, 'prefill_tps': int(f[2]) / (pms / 1000.0)
                     if pms else 0.0, 'finish': f[7]})
    totals = [s for s in stats if s.get('total')]
    arm = {'exe': exe, 'rows': rows, 'wall_s': round(time.time() - t0, 1)}
    if totals:
        t = totals[-1]
        n = max(1, int(t.get('tokens', t.get('token', 1))))
        alt = {'ssd': 'ssd_reads', 'vram': 'vram_hits', 'ram': 'ram_hits'}
        arm['tiers'] = {k: t.get(k, t.get(alt.get(k, k))) for k in NUM}
        arm['per_token'] = {k: round((t.get(k, t.get(alt.get(k, k))) or 0) / n, 2) for k in NUM if k != 'token'}
    if waits:
        arm['ssd_wait_ms_mean'] = round(statistics.mean(waits), 1)
        arm['ssd_wait_ms_p50'] = round(statistics.median(waits), 1)
    if profiles:
        # Only the decode tokens: a token-by-token prefill emits a line per prompt token too, and the last `reps x
        # decode` lines of a process are its decode tokens (the prefill comes first in every GEN request).
        tail = decode * reps
        warm = profiles[-tail:] if len(profiles) >= tail else profiles
        arm['profile_mean'] = {k: round(statistics.mean([x[k] for x in warm if k in x]), 2) for k in PHASES
                               if any(k in x for x in warm)}
        arm['profile_tokens'] = len(warm)
        arm['profile_total_s'] = sum(x.get('total', 0.0) for x in warm) / 1000.0
    # the per-layer dump can be huge; keep the tail in a file
    Path('bench_p6_engine.err').write_text('\n'.join(err[-4000:]), encoding='utf-8')
    return arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', action='append', required=True)
    ap.add_argument('--label', action='append', default=[])
    ap.add_argument('--arm-model', action='append', default=[],
                    help='the pack for one arm, in --exe order (repeatable; default: --model).  A candidate and its '
                         'baseline are different files measured in the same session.')
    ap.add_argument('--arm-args', action='append', default=[],
                    help='extra engine args for one arm, in --exe order (repeatable, spaces allowed)')
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompt', type=int, default=512)
    ap.add_argument('--decode', type=int, default=32)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--rounds', type=int, default=1, help='interleaved rounds of every arm')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--ids', help='use these real token ids (int32 .bin or text) as the prompt text instead of '
                                  'synthetic ids; the prompt is taken from the start and repeats if it is short')
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--json-out')
    ap.add_argument('engine_args', nargs='*')
    a = ap.parse_args()
    extra = [x for x in a.engine_args if x != '--']
    labels = a.label or [Path(e).parent.parent.name for e in a.exe]
    while len(labels) < len(a.exe):
        labels.append(Path(a.exe[len(labels)]).parent.parent.name)
    arm_args = [x.split() if x else [] for x in a.arm_args]
    while len(arm_args) < len(a.exe):
        arm_args.append([])
    models = list(a.arm_model)
    while len(models) < len(a.exe):
        models.append(a.model)
    source = read_ids(a.ids) if a.ids else synth_ids(256, a.seed)
    if a.ids:
        print(f'prompt text: {a.ids} ({len(source)} tokens)')
    ctx = a.prompt + a.decode + 8

    results: dict[str, list[dict]] = {label: [] for label in labels}
    for rnd in range(a.rounds):
        for k, (exe, label) in enumerate(zip(a.exe, labels)):
            print(f'round {rnd}: {label}', flush=True)
            results[label].append(run_arm(exe, models[k], ctx, a.prompt, a.decode, a.reps, source,
                                          extra + arm_args[k], a.timeout))

    print()
    head = (f'{"arm":>12} | {"round":>5} | {"decode tok/s":>12} | {"ms/tok":>7} | {"ssd/tok":>8} | '
            f'{"ssd MB/tok":>10} | {"ssd wait ms":>10} | {"skipped":>7}')
    print(head)
    print('-' * len(head))
    for label in labels:
        for i, arm in enumerate(results[label]):
            pt = arm.get('per_token', {})
            med = statistics.median(r['decode'] for r in arm['rows'][1:]) if len(arm['rows']) > 1 else \
                (arm['rows'][0]['decode'] if arm['rows'] else 0.0)
            print(f'{label:>12} | {i:5d} | {med:12.2f} | {1000.0 / med if med else 0:7.1f} | '
                  f'{pt.get("ssd", 0):8.1f} | {pt.get("ssd_bytes", 0) / 2**20:10.0f} | '
                  f'{arm.get("ssd_wait_ms_mean", -1):10.1f} | {pt.get("skipped", 0):7.1f}')
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({'args': extra, 'results': results}, indent=1), encoding='utf-8')
        print(f'wrote {a.json_out}')
    # The per-token phase breakdown, when the arms ran with --minimax-profile-time.
    if any(arm.get('profile_mean') for label in labels for arm in results[label]):
        print()
        head = f'{"arm":>12} | {"tok":>4} | ' + ' | '.join(f'{k:>10}' for k in PHASES if k != 'total')
        print(head)
        print('-' * len(head))
        for label in labels:
            for arm in results[label]:
                pm = arm.get('profile_mean')
                if not pm:
                    continue
                print(f'{label:>12} | {arm["profile_tokens"]:4d} | ' +
                      ' | '.join(f'{pm.get(k, 0):10.1f}' for k in PHASES if k != 'total'))
                print(f'{"":>12} |      | total {pm.get("total", 0):.1f} ms/token')
    return 0


if __name__ == '__main__':
    sys.exit(main())
