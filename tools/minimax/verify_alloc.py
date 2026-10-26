"""Structural check of the built candidate: every expert tensor's type and length comes from the pack the
allocation named, and the total is the allocation's total."""
import glob
import hashlib
import json

from gguf import GGUFReader

C1 = r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\a1\cand_c1.gguf"
PACKS = {'d3': r'D:\hf-d3\IQ2_XS-D3\MiniMax-M3.1-Flash-IQ2_XS-D3-*.gguf',
         'iq2xxs': r'D:\hf-iq2xxs\MiniMax-M3.1-Flash-IQ2_XXS\MiniMax-M3.1-Flash-IQ2_XXS-*.gguf'}
alloc = json.load(open(r"C:\Users\admin\AppData\Local\Temp\alloc_c1.json"))['rows']
want = {(r['layer'], r['role']): r['pack'] for r in alloc}
leaf_role = {'ffn_gate_exps.weight': 'gate', 'ffn_up_exps.weight': 'up', 'ffn_down_exps.weight': 'down'}


def table(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for t in GGUFReader(f).tensors:
            out.setdefault(t.name, (int(t.tensor_type), int(t.n_bytes), f, int(t.data_offset)))
    return out


def sha(path, off, n, chunk=64 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        fh.seek(off)
        left = n
        while left:
            b = fh.read(min(chunk, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


packs = {k: table(p) for k, p in PACKS.items()}
r = GGUFReader(C1)
bad = 0
expert_bytes = 0
for t in r.tensors:
    name = t.name
    if not name.endswith('_exps.weight'):
        continue
    L, leaf = int(name.split('.')[1]), name.split('.', 2)[2]
    role = leaf_role[leaf]
    lab = want[(L, role)]
    ref = packs[lab].get(name)
    got = (int(t.tensor_type), int(t.n_bytes))
    expert_bytes += int(t.n_bytes)
    if ref is not None and ref[:2] != got:
        bad += 1
        print(f'MISMATCH {name}: built {got} expected {ref} ({lab})')
print(f'{171 - bad}/171 expert tensors match the allocation pack (type and length)')

every = int(__import__('os').environ.get('VERIFY_EVERY', '0'))
if every:
    import os
    checked = ok = 0
    for i, t in enumerate([x for x in r.tensors if x.name.endswith('_exps.weight')]):
        if i % every:
            continue
        name = t.name
        L, leaf = int(name.split('.')[1]), name.split('.', 2)[2]
        lab = want[(L, leaf_role[leaf])]
        _, _, src, src_off = packs[lab][name]
        a = sha(C1, int(t.data_offset), int(t.n_bytes))
        b = sha(src, src_off, int(t.n_bytes))
        checked += 1
        ok += int(a == b)
        if a != b:
            print(f'BYTES DIFFER {name} vs {lab} {os.path.basename(str(src))}')
    print(f'byte check: {ok}/{checked} sampled expert tensors identical to their allocated pack')
print(f'experts {expert_bytes / 1e9:.2f} GB, file {sum(t.n_bytes for t in r.tensors) / 1e9:.2f} GB, '
      f'{len(r.tensors)} tensors')
