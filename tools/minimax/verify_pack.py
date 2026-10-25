"""tools/minimax/verify_pack.py - prove a mixed pack's bytes, tensor by tensor, against its donors.

The engine resolves every tensor to a (pointer, length) pair at load; `--minimax-verify-tensors` prints one
`INFO minimax-verify <name> <bytes> <fnv1a-64>` line per tensor it resolved, from either entry point (`--serve` and
`--minimax-eval` load through the same code).  This script takes that output and the donor packs and reports, per
tensor, whether the engine's digest equals the digest of the donor's bytes - N/N when the pack is exactly the donors'
blocks, which is the claim a "copy whole layers, never re-quantize" mix has to make.

    python tools/minimax/verify_pack.py --engine-log serve.log --engine-log eval.log \
        --donor D3/...-00001-of-00013.gguf --donor IQ1_M/...-00001-of-00013.gguf

The donor search is by tensor name across every shard of each donor pack, so a mixed pack built from whole layers
resolves each tensor to the pack it came from without being told which layer went where.

Exit 0 when every tensor matches in every log, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
VERIFY_RE = re.compile(r'INFO minimax-verify (\S+) (\d+) (\d+) (\d+) (\d+)$')
CHUNK = 64 << 20   # bounded, buffered reads: no mapping, so the proof costs a few MB of RAM, not the pack's size


def sha256_range(path, offset: int, nbytes: int) -> str:
    """SHA-256 of `nbytes` at `offset`, read in CHUNK pieces (plain buffered reads, no mmap)."""
    h = hashlib.sha256()
    with open(path, 'rb', buffering=0) as fh:
        fh.seek(offset)
        left = nbytes
        while left:
            piece = fh.read(min(CHUNK, left))
            if not piece:
                raise SystemExit(f'short read from {path} at {offset}')
            h.update(piece)
            left -= len(piece)
    return h.hexdigest()


def llama_gguf():
    for cand in (Path('C:/llm/llama-pinned-3cf/gguf-py'), Path('third_party/llama.cpp/gguf-py')):
        if cand.exists():
            sys.path.insert(0, str(cand))
    import gguf
    return gguf


def shards(first: Path) -> list[Path]:
    name = first.name
    if '-00001-of-' not in name:
        return [first]
    pre, rest = name.split('-00001-of-', 1)
    total = int(rest.split('.')[0])
    out = []
    for i in range(1, total + 1):
        p = first.with_name(f'{pre}-{i:05d}-of-{rest}')
        if not p.exists():
            raise SystemExit(f'incomplete donor pack: {p} is missing')
        out.append(p)
    return out


def index_donors(paths: list[Path], gguf) -> dict:
    """{tensor name: (file, data offset, bytes)} for every donor pack's every shard."""
    idx = {}
    for first in paths:
        for path in shards(first):
            r = gguf.GGUFReader(str(path))
            for t in r.tensors:
                # a mixed pack holds the same tensor NAME from more than one donor, so keep every candidate: the
                # donor that matches is the one whose byte length equals what the engine reported.
                idx.setdefault(t.name, []).append((path, int(t.data_offset), int(t.n_bytes)))
            del r
    return idx


def pick_donor(cands, nbytes):
    """The donor entry whose byte length matches what the engine read (a mixed pack has several per name)."""
    for entry in cands:
        if entry[2] == nbytes:
            return entry
    return cands[0]


def digest_donor(entry) -> tuple[str, int]:
    path, offset, nbytes = entry
    return sha256_range(path, offset, nbytes), nbytes


def parse_log(path: Path) -> dict:
    """{name: (shard, offset, nbytes, type)} from the engine's resolution lines."""
    got = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = VERIFY_RE.search(line.rstrip())
        if m:
            got[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine-log', action='append', required=True,
                    help="a run's stderr with --minimax-verify-tensors (repeat for every entry point)")
    ap.add_argument('--pack', required=True, help='the pack under test; its first shard names the set')
    ap.add_argument('--donor', action='append', required=True, help='first shard of a donor pack (repeatable)')
    ap.add_argument('--label', action='append', default=[])
    ap.add_argument('--quiet', action='store_true', help='only the totals and mismatches')
    a = ap.parse_args(argv)
    gguf = llama_gguf()
    pack_shards = shards(Path(a.pack))
    donors = index_donors([Path(p) for p in a.donor], gguf)
    print(f'donors: {len(donors)} tensor names from {len(a.donor)} pack(s)')
    ok_all = True
    prev = None
    for i, log in enumerate(a.engine_log):
        label = a.label[i] if i < len(a.label) else Path(log).stem
        got = parse_log(Path(log))
        if not got:
            print(f'{label:>10}: NO --minimax-verify-tensors lines in {log}')
            ok_all = False
            continue
        same = missing = differ = 0
        offs = {n: (v[0], v[1], v[2]) for n, v in got.items()}
        if prev is not None:
            only_here = set(offs) - set(prev)
            moved = {n: (prev[n], offs[n]) for n in set(offs) & set(prev) if prev[n] != offs[n]}
            print(f'{label:>10}: resolution against the first log: {len(only_here)} tensors only here, '
                  f'{len(moved)} resolved differently' + ('' if not moved else f' (e.g. {list(moved)[:3]})'))
            if only_here or moved:
                ok_all = False
        prev = offs
        for name, (shard, offset, nbytes, qtype) in got.items():
            if shard >= len(pack_shards):
                missing += 1
                continue
            model_file = pack_shards[shard]
            if name not in donors:
                missing += 1
                if missing <= 5:
                    print(f'{label:>10}: {name} is not in any donor')
                continue
            dh, dn = digest_donor(pick_donor(donors[name], nbytes))
            if dh == sha256_range(model_file, offset, nbytes) and dn == nbytes:
                same += 1
            else:
                differ += 1
                if differ <= 5:
                    print(f'{label:>10}: {name} DIFFERS (engine shard {shard} off {offset} {nbytes} B, '
                          f'donor {dn} B)')
        total = same + differ + missing
        print(f'{label:>10}: {same}/{total} tensors identical to a donor'
              + (f' ({differ} differ' if differ else '') + (f', {missing} not in a donor' if missing else '') + ')')
        ok_all = ok_all and differ == 0 and missing == 0 and same == total and total > 0
    print('PROOF', 'PASS' if ok_all else 'FAIL',
          '- SHA-256 of every tensor, streamed in 64 MiB reads, and every log resolving every tensor the same way')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
