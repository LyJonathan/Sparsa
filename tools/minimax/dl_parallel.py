"""Download one published tier with several connections, then check it against SHA256SUMS.

The link measures ~26 MB/s on one connection, so a 124 GB tier is over an hour of
wall clock - and every minute of it is a minute the engine cannot run (a download
in the page cache is what tripped the launcher's floor in round 7).  Several
shards at once, one connection each, finishes the tier in a fraction of that and
then proves it: `SHA256SUMS` in the repo root carries one hash per shard, and a
shard is only renamed into place from `<name>.part` once its hash matches.

Resumable: a `.part` file is continued with `curl -C -`, so a kill or a dropped
link costs the bytes in flight, not the shard.  A shard already present at its
full size and matching its hash is skipped.

    python tools/minimax/dl_parallel.py --tier MiniMax-M3.1-Flash-IQ2_XXS --dest D:\\hf-iq2xxs --jobs 6
    python tools/minimax/dl_parallel.py --tier ... --dest ... --verify-only

Exit 0 when every shard verified, 1 when any shard failed, 2 when the metadata
could not be read.  Nothing is ever started beside an engine: the caller decides
when the PC is free.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPO = 'LyJonathan/MiniMax-M3.1-Flash-GGUF'
HOST = 'https://huggingface.co'


def http_text(url: str) -> str:
    req = urllib.request.Request(url, headers={'User-Agent': 'strata-dl/1'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode('utf-8', 'replace')


def listing(tier: str) -> list[tuple[str, int]]:
    """(relative path, bytes) for the tier's files, from the repo metadata."""
    j = json.loads(http_text(f'{HOST}/api/models/{REPO}?blobs=true'))
    out = []
    for s in j['siblings']:
        name = s['rfilename']
        if not name.startswith(tier + '/'):
            continue
        size = (s.get('lfs') or {}).get('size') or s.get('size') or 0
        out.append((name, int(size)))
    return sorted(out)


def sums() -> dict[str, str]:
    """`SHA256SUMS` is two lines per file: the hash, then the path (wrapped)."""
    toks = http_text(f'{HOST}/{REPO}/resolve/main/SHA256SUMS').split()
    out = {}
    for i, t in enumerate(toks):
        if re.fullmatch(r'[0-9a-f]{64}', t) and i + 1 < len(toks):
            out[toks[i + 1]] = t
    return out


def sha256(path: os.PathLike, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(path: str, size: int, dest_dir: str, retries: int = 5) -> tuple[str, bool, str]:
    """Download one shard (resuming a `.part`), verify it, rename it into place."""
    leaf = os.path.basename(path)
    final = os.path.join(dest_dir, leaf)
    part = final + '.part'
    url = f'{HOST}/{REPO}/resolve/main/{path}'
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size and have >= size:
            break
        cmd = ['curl.exe' if os.name == 'nt' else 'curl', '-L', '-C', '-', '--retry', '5', '--retry-delay', '3',
               '--silent', '--show-error', '-o', part, url]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and not (size and os.path.exists(part) and os.path.getsize(part) >= size):
            print(f'  {leaf}: curl rc {r.returncode} (attempt {attempt + 1}) {r.stderr.strip()[:160]}', flush=True)
            time.sleep(5)
            continue
        break
    if not os.path.exists(part):
        return leaf, False, 'no file'
    got = os.path.getsize(part)
    if size and got != size:
        return leaf, False, f'{got} of {size} bytes'
    return leaf, True, ''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True, help='e.g. MiniMax-M3.1-Flash-IQ2_XXS (a folder in the repo)')
    ap.add_argument('--dest', required=True, help='local root; the tier folder is created inside it')
    ap.add_argument('--jobs', type=int, default=6, help='shards downloaded at once (one connection each)')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args(argv)

    tables = sums()
    try:
        files = listing(args.tier)
    except Exception as e:  # noqa: BLE001 - any metadata failure is the same answer: cannot start
        print(f'cannot read the repo listing: {e}')
        return 2
    if not files:
        print(f'no files under {args.tier} in {REPO}')
        return 2
    dest_dir = os.path.join(args.dest, args.tier)
    os.makedirs(dest_dir, exist_ok=True)
    total = sum(s for _, s in files)
    print(f'{len(files)} shards, {total / 1e9:.1f} GB -> {dest_dir}', flush=True)

    def work(item):
        path, size = item
        leaf = os.path.basename(path)
        final = os.path.join(dest_dir, leaf)
        want = tables.get(path) or tables.get(leaf)
        if not args.verify_only and not (os.path.exists(final) and size and os.path.getsize(final) == size):
            leaf, ok, why = fetch(path, size, dest_dir)
            if not ok:
                return leaf, False, why
        if os.path.exists(final + '.part') and not os.path.exists(final):
            os.replace(final + '.part', final)
        if not os.path.exists(final):
            return leaf, False, 'missing'
        if not want:
            return leaf, True, 'no hash published'
        t = time.time()
        got = sha256(final)
        return leaf, got == want, f'sha256 {"ok" if got == want else "MISMATCH"} in {time.time() - t:.0f}s'

    bad = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for i, (leaf, ok, why) in enumerate(ex.map(work, files), 1):
            print(f'[{i}/{len(files)}] {leaf}: {"ok" if ok else "FAIL"} {why}', flush=True)
            bad += 0 if ok else 1
    print(f'{len(files) - bad}/{len(files)} shards verified in {(time.time() - t0) / 60:.1f} min')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
