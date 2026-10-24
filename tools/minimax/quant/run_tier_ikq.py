#!/usr/bin/env python3
"""One machine's share of one MiniMax-M3.1-Flash tier: quantize, verify, upload, delete.

The four tiers this drives are the ik_llama.cpp ports and the lossless NVFP4 copy:

    IQ3_K   191.2 GB, experts GGML type 161
    IQ4_KS  233.5 GB, 162
    IQ2_KL  152.7 GB, 164
    NVFP4   246.1 GB, 40 (GGML_TYPE_NVFP4, experts verbatim plus a per-expert f32 divisor)

Every part is one contiguous layer range and the part index is fixed, so the parts from
different machines concatenate in layer order. Part 0 carries the KV metadata and the
globals; later parts carry only split.no / split.count / split.tensors.count.

Per part the driver does, in order: fetch the source shards this machine does not have
(rig-3 only), quantize, `part_verify.py` (header, split metadata, expert types, exact
size, sha256), upload to the public tier folder, re-read the uploaded file's LFS sha256
from the Hugging Face API and compare it with the local one, then delete the part.
Nothing is deleted before the upload matches. NVFP4 additionally runs
`verify_nvfp4_gguf.py` (max |diff| must be 0) before the upload.

    run_tier_ikq.py --plan --tier IQ3_K
    run_tier_ikq.py --machine rig-2 --tier IQ3_K
    run_tier_ikq.py --machine rig-3 --tier IQ4_KS --parts 3,5

Rates below are measured, per machine and per tier, with `strata-mm-quant --bench` on
layer 3, 8 experts of w1, at every logical processor (rig-2 24 threads, rig-3 16).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "LyJonathan/MiniMax-M3.1-Flash-GGUF"
UVX = os.path.expanduser("~/.local/bin/uvx")

TIERS = {
    "IQ3_K": {"recipe": "MM-IQ3_K", "expert": "IQ3_K", "gb": 191.230, "imatrix": True},
    "IQ4_KS": {"recipe": "MM-IQ4_KS", "expert": "IQ4_KS", "gb": 233.547, "imatrix": True},
    "IQ2_KL": {"recipe": "MM-IQ2_KL", "expert": "IQ2_KL", "gb": 152.680, "imatrix": True},
    "NVFP4": {"recipe": "MM-NVFP4", "expert": "NVFP4", "gb": 246.098, "imatrix": False},
}

SPLIT_COUNT = 13

# (part, machine, first layer, last layer) inclusive. Same layer ranges for every tier, so
# the parts of a tier concatenate in layer order. The split is by measured time: rig-2 is
# 1.4-1.6x rig-3 on these formats, so it owns 34 of the 57 MoE layers and rig-3 owns 23.
PARTS = (
    (0, "rig-2", 0, 4),
    (1, "rig-3", 5, 8),
    (2, "rig-2", 9, 14),
    (3, "rig-3", 15, 17),
    (4, "rig-2", 18, 22),
    (5, "rig-3", 23, 26),
    (6, "rig-2", 27, 31),
    (7, "rig-3", 32, 35),
    (8, "rig-2", 36, 40),
    (9, "rig-3", 41, 44),
    (10, "rig-2", 45, 49),
    (11, "rig-3", 50, 53),
    (12, "rig-2", 54, 59),
)

# Measured logical-f32 GB/s for the routed experts, `--bench` on layer 3, 8 experts of w1.
GBPS = {
    "IQ3_K": {"rig-2": 1.009, "rig-3": 0.720},
    "IQ4_KS": {"rig-2": 0.685, "rig-3": 0.439},
    "IQ2_KL": {"rig-2": 0.352, "rig-3": 0.220},
    "NVFP4": {"rig-2": 3.108, "rig-3": 2.031},
}
GBPS_NOTE = {
    "rig-2": "Ryzen 9 5900X, 24 threads, measured",
    "rig-3": "Ryzen 7 5700G, 16 threads, measured",
}
EXPERT_F32_GB = 1.652e12 / 1e9
SIDE_PER_LAYER_GB = (6.818 + 0.251 + 0.179 + 3.429 + 0.003) / 57
DENSE_FFN_PER_LAYER_GB = 0.722 / 3
GLOBALS_GB = 1.306 + 1.008 + 0.003


def cfg(machine):
    home = os.path.expanduser("~")
    base = os.path.join(home, "strata-work")
    root = os.path.join(base, "Strata-MiniMax")
    c = {
        "root": root,
        "exe": os.path.join(base, "build-mm-quant-engine", "strata-mm-quant"),
        "quant_dir": os.path.join(root, "tools", "minimax", "quant"),
        "imatrix": os.path.join(base, "imatrix-minimax-m3.1.gguf"),
        "log_dir": os.path.join(base, "logs"),
        "threads": 0,
    }
    if machine == "rig-2":
        # the full checkpoint, never deleted
        c["model"] = os.environ.get("MINIMAX_CHECKPOINT", os.path.join(home, "models", "MiniMax-M3.1-Flash"))
        c["fetch"] = False
    elif machine == "rig-3":
        c["model"] = os.path.join(base, "shards")
        c["fetch"] = True
    else:
        raise SystemExit("unknown machine " + machine)
    return c


def part_name(tier, part):
    return "MiniMax-M3.1-Flash-%s-%05d-of-%05d.gguf" % (tier, part + 1, SPLIT_COUNT)


def out_dir(tier):
    return os.path.join(os.path.expanduser("~"), "strata-work", "out", "MiniMax-M3.1-Flash-" + tier)


def moe_count(lo, hi):
    return sum(1 for layer in range(lo, hi + 1) if layer >= 3)


def per_moe_gb(tier):
    return (TIERS[tier]["gb"] - 57 * SIDE_PER_LAYER_GB - 3 * DENSE_FFN_PER_LAYER_GB - GLOBALS_GB) / 57


def projected_gb(tier, lo, hi):
    moe = moe_count(lo, hi)
    dense = sum(1 for layer in range(lo, hi + 1) if layer < 3)
    total = moe * per_moe_gb(tier) + (hi - lo + 1) * SIDE_PER_LAYER_GB + dense * DENSE_FFN_PER_LAYER_GB
    if lo == 0:
        total += GLOBALS_GB
    return total


def minutes_for(tier, machine, lo, hi):
    moe = moe_count(lo, hi)
    return moe * (EXPERT_F32_GB / 57) / GBPS[tier][machine] / 60.0


def plan_text(tier):
    t = TIERS[tier]
    rows = [r for r in PARTS]
    lines = []
    lines.append("%s. %d final split parts, experts %s." % (t["recipe"], SPLIT_COUNT, t["expert"]))
    lines.append("Output tensors about %.1f GB. Each part is <prefix>-%05d-of-%05d.gguf under "
                 "MiniMax-M3.1-Flash-%s/ on %s." % (t["gb"], 1, SPLIT_COUNT, tier, REPO))
    lines.append("KV metadata and globals (token_embd, output_norm, output) are only in part 0.")
    lines.append("Rigs never receive the checkpoint over the LAN; rig-2 reads its local copy, "
                 "rig-3 streams the shards a part needs from Hugging Face and deletes them after.")
    lines.append("")
    lines.append("part  %-52s  machine  layers   MoE   GB    minutes" % "file")
    totals = {m: {"moe": 0, "min": 0.0, "gb": 0.0} for m in GBPS[tier]}
    for part, machine, lo, hi in rows:
        moe = moe_count(lo, hi)
        minutes = minutes_for(tier, machine, lo, hi)
        gb = projected_gb(tier, lo, hi)
        totals[machine]["moe"] += moe
        totals[machine]["min"] += minutes
        totals[machine]["gb"] += gb
        lines.append("%4d  %-52s  %-6s  %2d-%-2d  %3d  %5.1f  %7.1f" % (
            part, part_name(tier, part), machine, lo, hi, moe, gb, minutes))
    lines.append("")
    for machine in ("rig-2", "rig-3"):
        lines.append("%-6s  %2d MoE layers  %5.1f GB  %5.1f min quant  (%.3f GB/s logical f32, %s)" % (
            machine, totals[machine]["moe"], totals[machine]["gb"], totals[machine]["min"],
            GBPS[tier][machine], GBPS_NOTE[machine]))
    makespan = max(totals[m]["min"] for m in totals)
    lines.append("quant wall-clock %.1f min (%.2f h), the slower machine; uploads overlap the next part." % (
        makespan, makespan / 60.0))
    return "\n".join(lines)


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run(cmd, log, env=None, cwd=None):
    log.write("CMD " + " ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    return proc.returncode


def remote_lfs(path_in_repo):
    """(sha256, size) of the uploaded file, from the public tree API. No token needed.

    The resolve URL's ETag is not the LFS oid once a file lives on Xet storage, so the
    repository tree is asked instead: its entries carry lfs.oid, which is the file's sha256.
    """
    folder = path_in_repo.split("/")[0]
    url = "https://huggingface.co/api/models/%s/tree/main/%s" % (REPO, folder)
    req = urllib.request.Request(url, headers={"User-Agent": "strata-mm-quant"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        entries = json.loads(resp.read().decode("utf-8"))
    for e in entries:
        if e.get("path") != path_in_repo:
            continue
        lfs = e.get("lfs") or {}
        oid = lfs.get("oid") or e.get("oid") or ""
        size = lfs.get("size") or e.get("size") or 0
        return oid, int(size)
    return "", 0


def main():
    ap = argparse.ArgumentParser(description="Run one machine's parts of one MiniMax tier")
    ap.add_argument("--tier", required=True, choices=sorted(TIERS))
    ap.add_argument("--machine", choices=("rig-2", "rig-3"))
    ap.add_argument("--parts", default="", help="comma-separated part indices; default every part of this machine")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-local", action="store_true", help="do not delete a part after it is verified on HF")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    tier = args.tier
    t = TIERS[tier]
    if args.plan:
        print(plan_text(tier))
        return 0
    if not args.machine:
        raise SystemExit("--machine is required")

    c = cfg(args.machine)
    os.makedirs(c["log_dir"], exist_ok=True)
    od = out_dir(tier)
    os.makedirs(od, exist_ok=True)
    log_path = os.path.join(c["log_dir"], "tier-%s-%s.log" % (tier, args.machine))
    log = open(log_path, "a", encoding="utf-8", buffering=1)

    order = [p for p, m, _lo, _hi in PARTS if m == args.machine]
    if args.parts:
        order = [int(x) for x in args.parts.split(",")]
    rows = {p: (lo, hi) for p, _m, lo, hi in PARTS}

    log.write("TIER_BEGIN %s %s %s parts=%s\n" % (tier, args.machine, stamp(), ",".join(map(str, order))))
    for index, part in enumerate(order):
        lo, hi = rows[part]
        name = part_name(tier, part)
        path = os.path.join(od, name)
        log.write("PART_BEGIN %d layers %d-%d %s\n" % (part, lo, hi, stamp()))
        started = time.time()

        # 1. source shards: rig-3 streams and deletes, rig-2 has the checkpoint
        if c["fetch"]:
            keep = []
            for later in order[index + 1:]:
                keep.append("%d-%d" % rows[later])
            cmd = [sys.executable, os.path.join(c["quant_dir"], "fetch_shards.py"),
                   "--dest", c["model"], "--layers", "%d-%d" % (lo, hi), "--sidecars"]
            if run(cmd, log) != 0:
                log.write("PART_FAIL %d fetch\n" % part)
                return 1

        # 2. quantize
        if not (os.path.isfile(path) and os.path.getsize(path) > 100 * 1024 * 1024):
            cmd = [c["exe"], "--model", c["model"],
                   "--recipe", os.path.join(c["quant_dir"], "recipes", t["recipe"] + ".json")]
            if t["imatrix"]:
                cmd.extend(["--imatrix", c["imatrix"]])
            cmd.extend(["--layers", "%d-%d" % (lo, hi),
                        "--split-part", str(part), "--split-count", str(SPLIT_COUNT),
                        "--out", os.path.join(od, "MiniMax-M3.1-Flash-" + tier)])
            if args.threads or c["threads"]:
                cmd.extend(["--threads", str(args.threads or c["threads"])])
            if args.dry_run:
                cmd.append("--dry-run")
            rc = run(cmd, log)
            if rc != 0:
                log.write("PART_FAIL %d quant rc=%d\n" % (part, rc))
                return rc
            if args.dry_run:
                log.write("PART_END %d dry-run %s\n" % (part, stamp()))
                continue
        else:
            log.write("SKIP existing %s (%d bytes)\n" % (path, os.path.getsize(path)))

        # 3. verify the file itself
        cmd = [sys.executable, os.path.join(c["quant_dir"], "part_verify.py"),
               "--gguf", path, "--part", str(part), "--count", str(SPLIT_COUNT),
               "--expert", t["expert"], "--json"]
        log.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log.write(proc.stdout.strip() + "\n")
        if proc.returncode != 0:
            log.write(proc.stderr.strip() + "\n")
            log.write("PART_FAIL %d verify\n" % part)
            return 1
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        sha = info["sha256"]
        size = info["size"]

        # 3b. NVFP4 must be lossless against the checkpoint, every matrix
        if tier == "NVFP4":
            cmd = [UVX, "--with", "numpy", "python3",
                   os.path.join(c["quant_dir"], "verify_nvfp4_gguf.py"), "--gguf", path,
                   "--model", c["model"], "--layers", "%d-%d" % (lo, hi)]
            rc = run(cmd, log)
            if rc != 0:
                log.write("PART_FAIL %d nvfp4-verify rc=%d\n" % (part, rc))
                return rc
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                tail = fh.read()[-4000:]
            if "max |GGUF dequant - spec A dequant| = 0.0" not in tail:
                log.write("PART_FAIL %d nvfp4-verify did not report 0.0\n" % part)
                return 1

        # 4. upload, then check the uploaded bytes' sha256 and size
        if not args.no_upload:
            dest = "MiniMax-M3.1-Flash-%s/%s" % (tier, name)
            cmd = [UVX, "--from", "huggingface_hub[hf_xet]", "hf", "upload", REPO, path, dest,
                   "--repo-type", "model"]
            rc = run(cmd, log)
            if rc != 0:
                log.write("UPLOAD_RETRY %s\n" % name)
                time.sleep(15)
                rc = run(cmd, log)
            if rc != 0:
                log.write("PART_FAIL %d upload rc=%d\n" % (part, rc))
                return rc
            got_sha, got_size = "?", 0
            for attempt in range(6):
                try:
                    got_sha, got_size = remote_lfs(dest)
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    log.write("REMOTE_RETRY %s %s\n" % (name, exc))
                    time.sleep(20)
            if got_sha != sha or got_size != size:
                log.write("PART_FAIL %d remote sha %s size %d, local %s %d\n" %
                          (part, got_sha, got_size, sha, size))
                return 1
            log.write("UPLOADED %s sha256=%s bytes=%d verified=ok\n" % (dest, sha, size))
            if not args.keep_local:
                os.remove(path)
                log.write("DELETED %s\n" % path)
        else:
            log.write("VERIFIED %s sha256=%s bytes=%d\n" % (path, sha, size))

        # 5. drop the source shards this part no longer needs (rig-3)
        if c["fetch"]:
            keep = [str(rows[later][0]) + "-" + str(rows[later][1]) for later in order[index + 1:]]
            cmd = [sys.executable, os.path.join(c["quant_dir"], "fetch_shards.py"),
                   "--dest", c["model"], "--layers", "%d-%d" % (lo, hi), "--delete"]
            if keep:
                cmd.extend(["--keep-layers", ",".join(keep)])
            run(cmd, log)

        log.write("PART_END %d %.1f s %s\n" % (part, time.time() - started, stamp()))
    log.write("TIER_DONE %s %s %s\n" % (tier, args.machine, stamp()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
