"""Prove the bytes the SSD tier ACTUALLY RETURNS, for experts from every source pack.

The bug this test exists for: the SSD tier opened its file by shard index against the
**base** pack, so an `--expert-source` layer's bytes were read from the wrong file.  The
resolution test (`test_expert_source.py`) proved where the loader *said* the tensor was; this
one proves what the read path *delivered*.

The engine gains `--minimax-verify-ssd-hash K`: after every unbuffered read it hashes the bytes
it just staged (SHA-256) and prints

    INFO minimax-ssd-hash <tensor> <expert> <bytes> <sha256> <flat shard index> <file leaf>

for the first K distinct experts of every expert tensor.  Here every expert is forced onto the
SSD tier (`--memory-limit-mib 600`, no VRAM slots) and the run is a real 4-token forward, so
the lines are the tier's own output.  Each line is compared with the SHA-256 of the same byte
range read straight from the pack the tensor was allocated to, at the offset the engine's
`--minimax-verify-tensors` table reports.

Two arms, both all-SSD:

  * the base pack alone, and
  * the base pack with layers 3-7 taken from the IQ1_M donor.

Requirements:
  1. every hashed expert matches its source pack byte for byte;
  2. the alt arm hashes experts from BOTH packs (a base layer and a moved layer);
  3. the moved block's digests differ between the two arms (a negative control: the check can
     tell two packs apart), and re-hashing one line at the next expert's offset must fail.

    python tools/minimax/test_ssd_source_hash.py

It starts the engine through `pc_launch.py` (the PC lock) and is skipped when a pack is absent.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = Path(os.environ.get("TEMP", r"C:\Users\admin\AppData\Local\Temp")) / "p16-ssd"
D3 = Path(r"D:\hf-d3\IQ2_XS-D3\MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf")
IQ1 = Path(r"D:\hf-iq1m\MiniMax-M3.1-Flash-IQ1_M\MiniMax-M3.1-Flash-IQ1_M-00001-of-00013.gguf")
EXE = ROOT / "build" / "strata-minimax.exe"
LAUNCH = ROOT / "tools" / "minimax" / "pc_launch.py"
BLOCK = (3, 7)
VERIFY_RE = re.compile(r"INFO minimax-verify (\S+) (\d+) (\d+) (\d+) (\d+)(?: (\S+))?$")
HASH_RE = re.compile(r"INFO minimax-ssd-hash (\S+) (-?\d+) (\d+) ([0-9a-f]{64}) (\d+) (\S+)$")


def shards(first: Path) -> list[Path]:
    name = first.name
    if "-00001-of-" not in name:
        return [first]
    pre, rest = name.split("-00001-of-", 1)
    total = int(rest.split(".")[0])
    return [first.with_name(f"{pre}-{i:05d}-of-{rest}") for i in range(1, total + 1)]


def run(tag: str, extra: list[str], no_launch: bool, timeout: int = 1800) -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    ids = TMP / "ids.txt"
    ids.write_text("1 2 3 4", encoding="utf-8")
    cmd = [str(EXE), "--model", str(D3), "--minimax-eval", str(ids),
           "--minimax-dump-logits", str(TMP / f"{tag}.npy"),
           "--max-context", "64", "--gpus", "0,1", "--minimax-tiers",
           "--minimax-vram-slots-per-layer", "0",
           # tiny RAM tier: every routed expert goes through the SSD read path under test
           "--memory-limit-mib", "600", "--cpu-threads", "8",
           # the batched prompt pass is the path a quality run reads experts through (`prefill=batch`,
           # reads=direct); without it the token path's queued reads are what run, and that path is not
           # instrumented yet
           "--minimax-prefill-batch", "512",
           "--minimax-verify-tensors", "--minimax-verify-ssd-hash", "1"] + extra
    launch = cmd
    if not no_launch:
        launch = [sys.executable, str(LAUNCH), "--owner", "maya", "--est-ram-gb", "25",
                  "--min-free-gb", "10", "--max-minutes", "45", "--"] + cmd
    with open(TMP / f"{tag}.log", "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.run(launch, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    print(f"{tag}: rc {p.returncode}")
    return p.returncode


def parse(log: Path) -> tuple[dict, list[tuple]]:
    tables, hashes = {}, []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = VERIFY_RE.search(line.rstrip())
        if m:
            tables[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(6) or "")
            continue
        h = HASH_RE.search(line.rstrip())
        if h:
            hashes.append((h.group(1), int(h.group(2)), int(h.group(3)), h.group(4), int(h.group(5)), h.group(6)))
    return tables, hashes


def sha_range(path: Path, offset: int, nbytes: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as fh:
        fh.seek(offset)
        left = nbytes
        while left:
            piece = fh.read(min(64 << 20, left))
            if not piece:
                raise SystemExit(f"short read from {path} at {offset}")
            h.update(piece)
            left -= len(piece)
    return h.hexdigest()


ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def check_arm(label: str, log: Path, flat: list[Path]) -> tuple[bool, dict]:
    text = log.read_text(encoding="utf-8", errors="replace")
    selftest = f"INFO minimax-ssd-hash-selftest abc {ABC_SHA256}"
    if selftest not in text:
        print(f"{label}: FAIL - the SHA-256 known-answer check is missing or wrong (expected '{selftest}')")
        return False, {}
    if "prefill=batch" not in text:
        print(f"{label}: FAIL - the run did not use the batched prompt path (prefill=batch), so the tier read "
              f"under test is not the one a quality run uses")
        return False, {}
    tables, hashes = parse(log)
    if not hashes:
        print(f"{label}: FAIL - no INFO minimax-ssd-hash lines (the tier never read, or the flag is not in this build)")
        return False, {}
    ok = 0
    bad = []
    per_file: dict[str, int] = {}
    seen = {}
    for name, expert, nbytes, digest, shard, leaf in hashes:
        if name not in tables:
            bad.append((name, "no resolution line"))
            continue
        tshard, offset, tbytes, tleaf = tables[name]
        if shard != tshard:
            bad.append((name, f"hash line shard {shard} != resolution shard {tshard}"))
            continue
        if tbytes % max(1, nbytes) != 0:
            bad.append((name, f"tensor bytes {tbytes} is not a multiple of the expert's {nbytes}"))
            continue
        file = flat[shard] if shard < len(flat) else None
        if file is None or file.name != leaf:
            bad.append((name, f"shard {shard} is {None if file is None else file.name}, line says {leaf}"))
            continue
        want = sha_range(file, offset + expert * nbytes, nbytes)
        seen[name] = digest
        per_file[leaf] = per_file.get(leaf, 0) + 1
        if want == digest:
            ok += 1
        else:
            bad.append((name, f"digest differs from {leaf} at {offset + expert * nbytes} (+{expert} experts, "
                              f"{tbytes // max(1, nbytes)} total)"))
    for name, why in bad[:8]:
        print(f"{label}: {name}: {why}")
    print(f"{label}: {ok}/{len(hashes)} SSD-tier byte ranges match their source pack "
          f"({', '.join(f'{k} x{v}' for k, v in sorted(per_file.items()))})")
    return ok == len(hashes) and not bad, seen


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-launch", action="store_true",
                    help="the caller already holds the PC lock: run the engine directly")
    args = ap.parse_args()
    for p in (D3, IQ1):
        if not p.exists():
            print(f"SKIP: {p} is not on this machine")
            return 0
    if not EXE.exists():
        print(f"SKIP: {EXE} is not built")
        return 0
    in_arm = run("base", [], args.no_launch)
    alt_arm = run("alt", ["--expert-source", f"{BLOCK[0]}-{BLOCK[1]}={IQ1}"], args.no_launch)
    if in_arm != 0 or alt_arm != 0:
        print("FAIL: an all-SSD arm did not run")
        return 1
    flat = shards(D3) + shards(IQ1)
    ok_base, base_seen = check_arm("base", TMP / "base.log", flat)
    ok_alt, alt_seen = check_arm("alt", TMP / "alt.log", flat)
    if not (ok_base and ok_alt):
        print("FAIL: the SSD tier returned bytes that are not their source pack's")
        return 1
    donor_leaf = IQ1.name
    moved = [n for n in alt_seen if BLOCK[0] <= int(n.split(".")[1]) <= BLOCK[1]]
    if not moved:
        print("FAIL: the alt arm never hashed a moved-block tensor (no evidence for the donor source)")
        return 1
    if not any(n not in moved for n in alt_seen):
        print("FAIL: the alt arm never hashed a base-pack tensor (no evidence for the base source)")
        return 1
    # negative control 1: the two arms must disagree on the moved block
    same = [n for n in moved if n in base_seen and base_seen[n] == alt_seen[n]]
    if same:
        print(f"FAIL: {len(same)} moved tensor(s) hash identically in both arms, e.g. {same[:3]}")
        return 1
    # negative control 2: hashing at the next expert's offset must not match
    tables, hashes = parse(TMP / "alt.log")
    name, expert, nbytes, digest, shard, leaf = next(h for h in hashes if h[0] in moved)
    file = flat[shard]
    offset = tables[name][1] + (expert + 1) * nbytes
    wrong = sha_range(file, offset, nbytes)
    if wrong == digest:
        print("FAIL: the next expert's bytes hash to the same value; the check cannot tell experts apart")
        return 1
    print(f"negative controls: {len(moved)} moved tensor(s) differ between arms; the next expert's range differs")
    print(f"PASS: {len(base_seen) + len(alt_seen)} SSD-tier reads across {len(set(base_seen) | set(alt_seen))} "
          f"tensors, from both the base pack and {donor_leaf}, hash to their own source pack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
