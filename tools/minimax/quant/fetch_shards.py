#!/usr/bin/env python3
"""Download MiniMax-M3.1-Flash shards for a layer range directly from Hugging Face.

The checkpoint is public. No token. Stdlib only, so a rig with python3 and no
curl/hf/pip can run it. Files that are already the right size are skipped.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "MiniMaxAI/MiniMax-M3.1-Flash"
BASE = "https://huggingface.co/" + REPO + "/resolve/main/"
INDEX_NAME = "model.safetensors.index.json"
SIDECARS = (
    "config.json",
    "generation_config.json",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
)
UA = "strata-mm-quant"


def parse_layers(text):
    layers = []
    if not text:
        return layers
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_s, hi_s = piece.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise SystemExit("empty layer range " + piece)
            layers.extend(range(lo, hi + 1))
        else:
            layers.append(int(piece))
    return layers


def request(url, extra_headers=None):
    headers = {"User-Agent": UA}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=120)


def remote_size(name):
    url = BASE + name
    with request(url, {"Range": "bytes=0-0"}) as resp:
        cr = resp.headers.get("Content-Range")
        if cr and "/" in cr:
            total = cr.rsplit("/", 1)[1]
            if total.isdigit():
                return int(total)
        length = resp.headers.get("Content-Length")
        if length and length.isdigit() and resp.status == 200:
            return int(length)
    raise SystemExit("could not read the size of " + name)


def download(name, dest_dir):
    dest = os.path.join(dest_dir, name)
    url = BASE + name
    try:
        total = remote_size(name)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("missing on HF (skipped):", name, flush=True)
            return 0
        raise
    have = os.path.getsize(dest) if os.path.isfile(dest) else 0
    if have == total:
        print("have", name, flush=True)
        return 0
    if have > total:
        print("replacing oversized", name, flush=True)
        have = 0
    headers = {}
    mode = "wb"
    if 0 < have < total:
        headers["Range"] = "bytes=%d-" % have
        mode = "ab"
    print("get", name, "have", have, "total", total, flush=True)
    tmp = dest + ".partial" if have == 0 else dest
    if have == 0 and os.path.isfile(tmp):
        os.remove(tmp)
    written = have
    try:
        with request(url, headers) as resp:
            if resp.status == 200 and have:
                # The server ignored Range. Start over.
                mode = "wb"
                written = 0
            with open(tmp if mode == "wb" and tmp != dest else dest, mode) as out:
                while True:
                    buf = resp.read(8 * 1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
                    written += len(buf)
    except Exception:
        print("download failed:", name, file=sys.stderr)
        raise
    if tmp != dest and os.path.isfile(tmp):
        os.replace(tmp, dest)
    if os.path.getsize(dest) != total:
        raise SystemExit("%s is %d bytes, expected %d" % (name, os.path.getsize(dest), total))
    print("got", name, flush=True)
    return total if have == 0 else total - have


def load_index(dest_dir):
    path = os.path.join(dest_dir, INDEX_NAME)
    if not (os.path.isfile(path) and os.path.getsize(path) > 1000):
        download(INDEX_NAME, dest_dir)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def shards_for_layers(weight_map, layers):
    needles = ["language_model.model.layers.%d." % n for n in layers]
    found = set()
    for tensor, shard in weight_map.items():
        for needle in needles:
            if needle in tensor:
                found.add(shard)
                break
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(description="Fetch MiniMax-M3.1-Flash shards from Hugging Face")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--layers", default="", help="3,4 or 0-4 or 0-4,14")
    ap.add_argument("--keep-layers", default="", help="with --delete, keep shards these layers still need")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--sidecars", action="store_true", help="also fetch config, tokenizer, chat template")
    ap.add_argument("--budget-gb", type=float, default=0.0, help="stop before a new shard would pass this many GB")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    layers = parse_layers(args.layers)
    weight_map = load_index(args.dest) if layers or args.list else None
    names = shards_for_layers(weight_map, layers) if layers else []
    if args.list:
        for name in names:
            print(name)
        return 0
    if args.sidecars:
        for name in SIDECARS:
            download(name, args.dest)
    fetched = 0
    for name in names:
        size = remote_size(name)
        if args.budget_gb > 0 and fetched + size > args.budget_gb * 1e9:
            print("budget %.1f GB reached before %s" % (args.budget_gb, name), flush=True)
            break
        path = os.path.join(args.dest, name)
        if os.path.isfile(path) and os.path.getsize(path) == size:
            print("have", name, flush=True)
            fetched += size
            continue
        download(name, args.dest)
        fetched += size
    if args.delete:
        keep = set(shards_for_layers(weight_map, parse_layers(args.keep_layers))) if args.keep_layers else set()
        for name in names:
            if name in keep:
                print("keep", name, flush=True)
                continue
            path = os.path.join(args.dest, name)
            if os.path.isfile(path):
                os.remove(path)
                print("deleted", name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
