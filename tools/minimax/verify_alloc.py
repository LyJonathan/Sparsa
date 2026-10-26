"""Structural + byte proof of a built candidate: every expert tensor's type and length comes from the pack the
allocation named, and (with `--all`, the default) every expert tensor's SHA-256 equals the same byte range of its
source pack.

    python tools/minimax/verify_alloc.py --pack _local/a1/cand_c1.gguf --alloc alloc_c1.json \
        --packs d3='D:\\hf-d3\\...\\*-00001-of-*' --packs iq2xxs='D:\\hf-iq2xxs\\...'

`--every N` samples instead (the old default was every 7th); `--every 1` and `--all` check all 171 tensors.
`--threads` hashes several tensors at once; the disk read is the limit on this box.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROLE = {'ffn_gate_exps.weight': 'gate', 'ffn_up_exps.weight': 'up', 'ffn_down_exps.weight': 'down'}


def llama_gguf():
    for cand in (Path('C:/llm/llama-pinned-3cf/gguf-py'), Path('third_party/llama.cpp/gguf-py')):
        if cand.exists():
            sys.path.insert(0, str(cand))
    import gguf
    return gguf


def pack_table(gguf, pattern: str) -> dict:
    """{tensor name: (type, n_bytes, file, data offset)} over every shard of a pack."""
    out = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'no pack files match {pattern!r}')
    for f in files:
        r = gguf.GGUFReader(str(f))
        for t in r.tensors:
            out.setdefault(t.name, (int(t.tensor_type), int(t.n_bytes), f, int(t.data_offset)))
        del r
    return out, files


def sha_range(path: str, off: int, n: int, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        fh.seek(off)
        left = n
        while left:
            b = fh.read(min(chunk, left))
            if not b:
                raise SystemExit(f'short read from {path} at {off}')
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pack', default=r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\a1\cand_c1.gguf")
    ap.add_argument('--alloc', default=os.path.join(os.environ.get('TEMP', r'C:\Users\admin\AppData\Local\Temp'),
                                                    'alloc_c1.json'))
    ap.add_argument('--packs', action='append', required=True, metavar='LABEL=GLOB',
                    help='a source pack the allocation names (repeatable)')
    ap.add_argument('--all', action='store_true', help='hash every expert tensor (default unless --every is given)')
    ap.add_argument('--every', type=int, default=0, help='sample every N-th expert tensor (0 = all)')
    ap.add_argument('--threads', type=int, default=4, help='tensors hashed concurrently')
    ap.add_argument('--out', default=None, help='write the per-tensor result as JSON')
    a = ap.parse_args(argv)

    gguf = llama_gguf()
    alloc = json.loads(Path(a.alloc).read_text(encoding='utf-8'))['rows']
    want = {(int(r['layer']), r['role']): r['pack'] for r in alloc}
    labels = [s.partition('=')[0] for s in a.packs]
    packs = {}
    for spec in a.packs:
        label, _, pattern = spec.partition('=')
        packs[label], files = pack_table(gguf, pattern)
        print(f'{label}: {len(files)} shard(s), {len(packs[label])} tensors from {pattern}', flush=True)

    started = time.time()
    r = gguf.GGUFReader(a.pack)
    experts = [t for t in r.tensors if t.name.endswith('_exps.weight')]
    bad = []

    def check(t):
        name = t.name
        layer = int(name.split('.')[1])
        role = ROLE[name.split('.', 2)[2]]
        label = want[(layer, role)]
        ref = packs[label].get(name)
        got = (int(t.tensor_type), int(t.n_bytes))
        if ref is None:
            return {'name': name, 'pack': label, 'ok': False, 'why': 'missing from its source pack'}
        if ref[:2] != got:
            return {'name': name, 'pack': label, 'ok': False,
                    'why': f'type/length {got} != {ref[:2]} in {label}'}
        return {'name': name, 'pack': label, 'ok': True, 'type': got[0], 'bytes': got[1],
                'src': ref[2], 'src_off': ref[3], 'offset': int(t.data_offset)}

    rows = [check(t) for t in experts]
    structural = sum(1 for x in rows if x['ok'])
    print(f'{structural}/{len(rows)} expert tensors match the allocation pack (type and length)', flush=True)

    step = 1 if a.all or not a.every else a.every
    todo = [x for i, x in enumerate(rows) if x['ok'] and i % step == 0]

    def digest(x):
        return x['name'], sha_range(a.pack, x['offset'], x['bytes']), sha_range(x['src'], x['src_off'], x['bytes'])

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, a.threads)) as ex:
        for name, built, src in ex.map(digest, todo):
            if built == src:
                ok += 1
            else:
                x = next(y for y in todo if y['name'] == name)
                bad.append({'name': name, 'pack': x['pack'], 'why': 'SHA-256 differs from the source pack'})
                print(f'BYTES DIFFER {name} vs {x["pack"]} {os.path.basename(x["src"])}', flush=True)

    expert_gb = sum(x['bytes'] for x in rows) / 1e9
    file_gb = sum(int(t.n_bytes) for t in r.tensors) / 1e9
    print(f'byte check: {ok}/{len(todo)} expert tensors identical (SHA-256) to the pack they were allocated from'
          + (f' - {len(bad)} DIFFER' if bad else ''), flush=True)
    print(f'experts {expert_gb:.2f} GB, file {file_gb:.2f} GB, {len(r.tensors)} tensors, '
          f'{time.time() - started:.0f}s', flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({'pack': a.pack, 'allocation': a.alloc, 'rows': rows,
                                           'bad': bad, 'checked': len(todo), 'identical': ok,
                                           'expert_gb': round(expert_gb, 3), 'file_gb': round(file_gb, 3)},
                                          indent=1), encoding='utf-8')
    good = structural == len(rows) and ok == len(todo) and not bad
    print('PROOF', 'PASS' if good else 'FAIL')
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
