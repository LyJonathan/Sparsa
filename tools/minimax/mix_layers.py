"""Build a RAM-fit MiniMax-M3.1-Flash GGUF by mixing whole layers from two published packs.

A1 of the Maya round: the D3 pack is 138 GiB and the box has 128 GB, so ~15 routed experts a token come off the
NVMe.  GGUF stores all 128 experts of a layer in one tensor, so the cheapest mix that stays a *standard* GGUF is
**per layer**: each MoE layer's `ffn_*_exps` tensors are copied whole, bit for bit, from either pack, and every
other tensor (dense, attention, embeddings, `lm_head`, norms) comes from the reference pack.  Nothing is
re-quantized, so a layer taken from the cold pack is byte-identical to the cold pack's layer.

    python tools/minimax/mix_layers.py --hot D3/...-00001-of-00013.gguf --cold IQ1_M/...-00001-of-00013.gguf \
        --out D:/hf-mix/mix.gguf --cold-frac 0.44 --order deep --profile D:/hf-d3/profiles/mm3-d3.bin

`--cold-frac` is the share of MoE layers taken from the cold pack.  `--order mass` (default) takes the layers with
the least routing mass (from the STRP profile) first, so the layers that carry the conversation keep their bits;
`--order deep` takes the deepest layers first, because a decoder's late layers tolerate fewer bits (Project Maya's
TRUNK_KEEP map: layers 3-7 KL 0.040, 23-29 0.012, 30-44 0.002-0.003).

The writer is a byte splice, not gguf-py's: the source's key/value block is copied **verbatim** (60 keys, every GGUF
value type) and only the tensor-info block is rebuilt, because the tensors' types and offsets change while their
names and dims do not.  gguf-py's writer cannot copy a key/value block type-blindly, and re-encoding 60 mixed-type
keys is exactly where a silent bug would live.
"""
from __future__ import annotations

import argparse
import glob
import json
import struct
import sys
from pathlib import Path


def llama_gguf():
    for cand in (Path('C:/llm/llama-pinned-3cf/gguf-py'),):
        if cand.exists():
            sys.path.insert(0, str(cand))
    import gguf
    return gguf


def shards(first: Path) -> list[Path]:
    """The other shards beside `first` (same stem up to -NNNNN-of-MMMMM)."""
    name = first.name
    if '-00001-of-' not in name:
        return [first]
    pre, rest = name.split('-00001-of-', 1)
    total = int(rest.split('.')[0])
    out = [first]
    for i in range(2, total + 1):
        p = first.with_name(f'{pre}-{i:05d}-of-{rest}')
        if not p.exists():
            raise SystemExit(f'pack is incomplete: {p} is missing')
        out.append(p)
    return out


def index_pack(first: Path, gguf) -> dict:
    """{tensor name: (file, ggml type id, GGUF-order shape, data offset, bytes)}."""
    idx = {}
    for path in shards(first):
        r = gguf.GGUFReader(str(path))
        for t in r.tensors:
            idx[t.name] = (path, int(t.tensor_type), tuple(int(x) for x in t.shape),
                           int(t.data_offset), int(t.n_bytes))
        del r
    return idx


_KV_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_kv_region(path: Path) -> tuple[bytearray, bytes, int]:
    """(the 24-byte header, the key/value block to write, the key count to write).

    The block is copied byte for byte, minus the `split.*` bookkeeping: the source is a 13-shard pack and the output
    is one file, so `split.no` / `split.count` / `split.tensors.count` must go (the engine's GgufModel treats a file
    that declares `split.count > 1` as shard 1 of that many).  Both source packs spell them `u16`/`u16`/`i32`, but
    nothing here assumes that: the whole key/value entry is dropped by its span, so the type does not matter.
    """
    raw = Path(path).read_bytes()
    if raw[:4] != b'GGUF':
        raise SystemExit(f'{path}: not a GGUF')
    version = struct.unpack_from('<I', raw, 4)[0]
    if version != 3:
        raise SystemExit(f'{path}: GGUF version {version}, expected 3')
    o = 24
    kv_count = struct.unpack_from('<Q', raw, 16)[0]
    kept = bytearray()
    n_kept = 0
    all_keys = []
    for _ in range(kv_count):
        start = o
        n = struct.unpack_from('<Q', raw, o)[0]
        key = raw[o + 8:o + 8 + n].decode('utf-8', 'replace')
        o += 8 + n
        vt = struct.unpack_from('<I', raw, o)[0]
        o += 4
        if vt in _KV_FIXED:
            o += _KV_FIXED[vt]
        elif vt == 8:
            n = struct.unpack_from('<Q', raw, o)[0]
            o += 8 + n
        elif vt == 9:
            sub, cnt = struct.unpack_from('<IQ', raw, o)
            o += 12
            for _ in range(cnt):
                if sub in _KV_FIXED:
                    o += _KV_FIXED[sub]
                elif sub == 8:
                    n = struct.unpack_from('<Q', raw, o)[0]
                    o += 8 + n
                else:
                    raise SystemExit(f'{path}: array of value type {sub} is not known')
        else:
            raise SystemExit(f'{path}: value type {vt} is not known')
        all_keys.append(key)
        if key.startswith('split.'):
            continue
        kept += raw[start:o]
        n_kept += 1
    dropped = [k for k in all_keys if k.startswith('split.')]
    if dropped:
        print(f'dropping {len(dropped)} shard key(s): {", ".join(dropped)}', flush=True)
    return bytearray(raw[0:24]), bytes(kept), n_kept


def profile_mass(path: str, moe_layers: list[int]) -> dict:
    """Per-MoE-layer routing mass from an STRP profile: `STRP`, version, n_layers, n_expert, slots, n_ranked
    (int32), then n_ranked (uint16 layer, uint16 expert) pairs by descending routing frequency.  A layer's mass is
    the sum of (n_ranked - rank) over its experts there."""
    vals = {li: 0.0 for li in moe_layers}
    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError):
        return vals
    if not raw.startswith(b'STRP') or len(raw) < 24:
        return vals
    _ver, n_layers, _n_expert, _slots, n_ranked = struct.unpack_from('<iiiii', raw, 4)
    if n_ranked <= 0 or n_layers <= 0:
        return vals
    at = 24
    for r in range(n_ranked):
        if at + 4 > len(raw):
            break
        li, _e = struct.unpack_from('<HH', raw, at)
        at += 4
        if li in vals:
            vals[li] += float(n_ranked - r)
    return vals


def pack_str(s: str) -> bytes:
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hot', required=True, help='the reference pack (dense tensors and the warm layers)')
    ap.add_argument('--cold', required=True, help='the pack the coldest layers come from (the lowest bits)')
    ap.add_argument('--mid', help='a middle band pack (e.g. IQ2_XXS) taken by the next-coldest layers')
    ap.add_argument('--mid-frac', type=float, default=0.0, help='share of MoE layers from the middle band')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cold-frac', type=float, default=0.44, help='share of MoE layers from the cold pack')
    ap.add_argument('--order', default='mass', choices=('mass', 'deep', 'shallow'))
    ap.add_argument('--cold-layers', help='explicit MoE layers to take from the cold pack, "35,36,37" (overrides '
                                          '--cold-frac/--order), which is how a sensitivity sweep names one block')
    ap.add_argument('--profile', help='STRP profile: its per-layer routing mass orders the cold layers')
    ap.add_argument('--alloc', help='JSON from a1_dp.py: take each expert tensor from the pack its row names '
                                    '(per layer AND projection; needs --pack)')
    ap.add_argument('--pack', action='append', default=[],
                    help='label=glob, repeatable: the packs --alloc may name (the label must match the JSON)')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and the sizes, write nothing')
    a = ap.parse_args(argv)

    gguf = llama_gguf()
    hot_first, cold_first = Path(a.hot), Path(a.cold)
    hot = index_pack(hot_first, gguf)
    cold = index_pack(cold_first, gguf)
    mid = index_pack(Path(a.mid), gguf) if a.mid else {}
    print(f'hot  {hot_first.name}: {len(hot)} tensors, {len(shards(hot_first))} shards')
    print(f'cold {cold_first.name}: {len(cold)} tensors, {len(shards(cold_first))} shards')

    moe_layers = sorted({int(n.split('.')[1]) for n in hot
                         if n.startswith('blk.') and n.endswith('_exps.weight')})
    missing = [n for n in hot if n not in cold]
    if missing:
        print(f'note: {len(missing)} tensors exist only in the hot pack (kept from hot)', flush=True)

    # a per-tensor allocation (57 layers x 3 projections) from a1_dp.py: each expert tensor comes from the pack its
    # row names.  This is the only way to express a GGUF-level assignment: one tensor carries one ggml type, so the
    # allocation cannot be finer than (layer, projection) and cannot be coarser either.
    alloc_packs: dict[str, dict] = {}
    alloc: dict[tuple[int, str], str] = {}
    if a.alloc:
        role_leaf = {'gate': 'ffn_gate_exps.weight', 'up': 'ffn_up_exps.weight', 'down': 'ffn_down_exps.weight'}
        for spec in a.pack:
            label, _, pattern = spec.partition('=')
            first = Path(sorted(glob.glob(pattern))[0] if any(c in pattern for c in '*?') else pattern)
            alloc_packs[label] = index_pack(first, gguf)
            print(f'alloc pack {label}: {first.name}: {len(alloc_packs[label])} tensors')
        if not alloc_packs:
            raise SystemExit('--alloc needs at least one --pack label=glob')
        j = json.loads(Path(a.alloc).read_text(encoding='utf-8'))
        for r in j['rows']:
            alloc[(int(r['layer']), role_leaf[r['role']])] = r['pack']
        unknown = {lab for lab in alloc.values() if lab not in alloc_packs}
        if unknown:
            raise SystemExit(f'--alloc names packs with no --pack: {sorted(unknown)}')
        print(f'alloc: {len(alloc)} expert tensors assigned')

    def layer_bytes(idx, li):
        return sum(idx[n][4] for n in idx if n.startswith(f'blk.{li}.') and n.endswith('_exps.weight'))

    hot_per = sum(layer_bytes(hot, li) for li in moe_layers) / len(moe_layers)
    cold_per = sum(layer_bytes(cold, li) for li in moe_layers) / len(moe_layers)

    mass = profile_mass(a.profile, moe_layers) if a.profile else {li: 0.0 for li in moe_layers}
    if a.order == 'deep':
        rank = sorted(moe_layers, key=lambda li: -li)
    elif a.order == 'shallow':
        rank = sorted(moe_layers)
    else:
        rank = sorted(moe_layers, key=lambda li: (mass[li], -li))
    if alloc:
        # every expert tensor comes from its own row; no layer-level sets are in play
        cold_layers, mid_layers, n_cold, n_mid = set(), set(), 0, 0
    elif a.cold_layers:
        cold_layers = {int(x) for x in a.cold_layers.replace(' ', '').split(',') if x != ''}
        unknown = cold_layers - set(moe_layers)
        if unknown:
            raise SystemExit(f'--cold-layers names layers with no experts in the hot pack: {sorted(unknown)}')
        n_cold = len(cold_layers)
        rank = sorted(moe_layers, key=lambda li: (0 if li in cold_layers else 1, li))
    else:
        n_cold = max(0, min(len(moe_layers), int(round(a.cold_frac * len(moe_layers)))))
        cold_layers = set(rank[:n_cold])
    n_mid = max(0, min(len(moe_layers) - n_cold, int(round(a.mid_frac * len(moe_layers))))) if a.mid else 0
    mid_layers = set(rank[n_cold:n_cold + n_mid]) if n_mid else set()

    def src_idx(li, leaf=''):
        if alloc and leaf:
            lab = alloc.get((li, leaf))
            if lab:
                return alloc_packs[lab]
        return cold if li in cold_layers else (mid if li in mid_layers else hot)
    expert_total = sum(layer_bytes(src_idx(li), li) for li in moe_layers)
    if alloc:
        expert_total = sum(sum(src_idx(li, leaf)[f'blk.{li}.{leaf}'][4] for leaf in
                               ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight'))
                           for li in moe_layers)
    other_total = sum(v[4] for k, v in hot.items() if not k.endswith('_exps.weight'))
    mass_cold = 100.0 * sum(mass[li] for li in cold_layers) / max(1e-9, sum(mass.values())) if a.profile else 0.0
    print(f'MoE layers {len(moe_layers)}, {n_cold} coldest ({100.0 * n_cold / len(moe_layers):.1f}%, '
          f'{mass_cold:.1f}% of the profile mass)' + (f', {n_mid} middle' if n_mid else ''))
    print(f'experts: hot {hot_per / 1e9:.3f} GB/layer, cold {cold_per / 1e9:.3f} -> {expert_total / 1e9:.1f} GB')
    print(f'other tensors {other_total / 1e9:.2f} GB -> file ~{(expert_total + other_total) / 1e9:.1f} GB')
    print('coldest layers:', ','.join(str(x) for x in sorted(cold_layers)))
    if n_mid:
        print('middle layers:', ','.join(str(x) for x in sorted(mid_layers)))
    if a.dry_run:
        return 0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hdr, kv_bytes, kv_count = read_kv_region(hot_first)
    alignment = 32
    infos = bytearray()
    data_off = 0
    plan = []
    order = [n for n in hot if n in cold] + [n for n in hot if n not in cold]
    for name in order:
        li = int(name.split('.')[1]) if name.startswith('blk.') else -1
        use_cold = name.endswith('_exps.weight') and li in cold_layers
        use_mid = name.endswith('_exps.weight') and li in mid_layers
        if alloc and name.endswith('_exps.weight'):
            leaf = name.split('.', 2)[2]
            src = alloc_packs[alloc[(li, leaf)]].get(name) or hot[name]
            use_cold = alloc[(li, leaf)] != 'd3'
        else:
            src = (cold if use_cold else mid if use_mid else hot).get(name) or hot[name]
        path, qtype, shape, offset, nbytes = src
        infos += pack_str(name)
        infos += struct.pack('<I', len(shape))
        for d in shape:
            infos += struct.pack('<Q', d)
        infos += struct.pack('<I', qtype)
        infos += struct.pack('<Q', data_off)
        plan.append((name, path, offset, nbytes, use_cold))
        data_off += nbytes + ((-nbytes) % alignment)
    struct.pack_into('<Q', hdr, 8, len(plan))     # tensor count
    struct.pack_into('<Q', hdr, 16, kv_count)     # key/value count: the dropped split.* keys must come off this
    # the padding aligns the ABSOLUTE position of the data, not the length of the info block: the key/value block
    # does not end on an alignment boundary (it ended 8 bytes in on this pack), so `(-len(infos)) % 32` alone put the
    # data 8 bytes past where the table says it starts and every tensor read one expert's tail.
    pad = (-(len(hdr) + len(kv_bytes) + len(infos))) % alignment
    total = len(hdr) + len(kv_bytes) + len(infos) + pad + data_off
    print(f'writing {out} ({total / 1e9:.1f} GB, {len(plan)} tensors, {kv_count} keys)', flush=True)
    done = 0
    with open(out, 'wb') as fh:
        fh.write(hdr)
        fh.write(kv_bytes)
        fh.write(infos)
        fh.write(bytes(pad))
        for name, path, offset, nbytes, use_cold in plan:
            with open(path, 'rb', buffering=0) as src:
                src.seek(offset)
                left = nbytes
                while left:
                    chunk = src.read(min(1 << 24, left))
                    if not chunk:
                        raise SystemExit(f'short read on {name} from {path}')
                    fh.write(chunk)
                    left -= len(chunk)
            fh.write(bytes((-nbytes) % alignment))
            done += nbytes
            if done % (16 << 30) < nbytes:
                print(f'  {done / 1e9:.0f}/{total / 1e9:.0f} GB', flush=True)
    print(f'wrote {out} ({done / 1e9:.1f} GB of tensor data)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
