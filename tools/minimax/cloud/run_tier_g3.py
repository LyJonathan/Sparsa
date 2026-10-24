"""One G3 tier on a rented Linux box: download, verify, measure, report, delete.

The box needs the engine built (tools/minimax/build_linux_g3.sh), the FP32 reference at
<ref>/logits + <ref>/ids (see README.md in this folder), and an HF token-free model repo.

    python3 tools/minimax/cloud/run_tier_g3.py --tier IQ1_M --work /work --exe /work/build/strata-minimax

Steps, in order:
  1. read SHA256SUMS from the HF repo and refuse to start unless all 13 lines for the tier exist
     (a tier that is still uploading has fewer);
  2. hf download with 8 workers (hf_xet: ~480-600 MiB/s on molab);
  3. sha256 every part against SHA256SUMS (13 x ~11 GiB in ~20 s with 8 threads);
  4. g3_sparsa.py on the 6 short prompts, flags below;
  5. copy results.json / results.md / run.log to <out>, then delete the tier's GGUF files.

--keep leaves the GGUF files in place (for a second pass on the same tier).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "LyJonathan/MiniMax-M3.1-Flash-GGUF"
FOLDER = "MiniMax-M3.1-Flash-{tier}"
PARTS = 13
CUDA_LIB = "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib"


def log(msg):
    print(msg, flush=True)


def sums_for(tier):
    url = "https://huggingface.co/%s/resolve/main/SHA256SUMS" % REPO
    txt = urllib.request.urlopen(url, timeout=60).read().decode()
    want = {}
    for line in txt.splitlines():
        if "MiniMax-M3.1-Flash-%s/" % tier not in line:
            continue
        digest, path = line.split(None, 1)
        want[os.path.basename(path.strip())] = digest.strip()
    return want


def hf_folder_size(tier):
    api = "https://huggingface.co/api/models/%s/tree/main/%s" % (REPO, FOLDER.format(tier=tier))
    files = json.load(urllib.request.urlopen(api, timeout=60))
    return (len([f for f in files if f["type"] == "file"]),
            sum(f.get("size", 0) for f in files if f["type"] == "file") / 1024 ** 3)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--work", default="/work")
    ap.add_argument("--ref", default=None, help="reference root (default <work>/g3ref)")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--cpu-threads", default="20")
    ap.add_argument("--memory-limit-mib", default="140000")
    ap.add_argument("--vram-slots", default="0",
                    help="0 disables the VRAM expert tier, which is what the rig-2 table used; "
                         "a nonzero value adds a GPU expert tier but measured no decode gain on molab")
    ap.add_argument("--prefill-experts", default="cpu",
                    help="cpu for the ik formats: gpu prefill fails on type 164 (IQ2_KL)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    tier = args.tier
    work = args.work
    ref = args.ref or os.path.join(work, "g3ref")
    exe = args.exe or os.path.join(work, "build", "strata-minimax")
    out = args.out or os.path.join(work, "out", tier)
    folder = os.path.join(work, "hf", FOLDER.format(tier=tier))

    want = sums_for(tier)
    nfiles, gib = hf_folder_size(tier)
    log("HF %s: %d files, %.1f GiB; SHA256SUMS lines for %s: %d" % (tier, nfiles, gib, tier, len(want)))
    if len(want) < PARTS:
        raise SystemExit("tier %s is not complete on HF (%d/%d sha256 lines); skipping"
                         % (tier, len(want), PARTS))

    if not args.skip_download:
        t0 = time.time()
        subprocess.run(["hf", "download", REPO, "--include", FOLDER.format(tier=tier) + "/*",
                        "--local-dir", os.path.join(work, "hf"), "--max-workers", "8"], check=True)
        log("download %.0fs" % (time.time() - t0))

    parts = sorted(glob.glob(os.path.join(folder, "*.gguf")))
    if len(parts) != PARTS:
        raise SystemExit("expected %d parts, found %d in %s" % (PARTS, len(parts), folder))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for name, got in zip([os.path.basename(p) for p in parts], ex.map(sha256, parts)):
            if want.get(name) != got:
                raise SystemExit("sha256 mismatch: %s got %s want %s" % (name, got, want.get(name)))
    log("sha256 ok for %d parts in %.0fs" % (len(parts), time.time() - t0))

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = CUDA_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g3_sparsa.py"),
           "--model", parts[0], "--out", out, "--tier", "MM-" + tier, "--exe", exe,
           "--ref", os.path.join(ref, "logits"), "--ids", os.path.join(ref, "ids"),
           "--gpus", args.gpus, "--layer-split", "auto", "--kv", "vram",
           "--prefill-batch", "512", "--prefill-experts", args.prefill_experts, "--prefill-dense", "f32",
           "--vram-slots", args.vram_slots, "--vram-reserve-mib", "700",
           "--memory-limit-mib", args.memory_limit_mib, "--max-context", "8192",
           "--cpu-threads", args.cpu_threads, "--no-ppl"]
    log("CMD " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    log(p.stdout)
    if p.stderr:
        log("---- stderr tail ----\n" + p.stderr[-4000:])
    open(os.path.join(out, "run.log"), "w", encoding="utf-8").write(p.stdout + "\n---- stderr ----\n" + p.stderr)
    log("rc %d wall %.0fs (%.1f min)" % (p.returncode, wall, wall / 60))
    if p.returncode != 0:
        raise SystemExit("g3 failed for " + tier)

    if not args.keep:
        t0 = time.time()
        for path in parts:
            os.remove(path)
        log("deleted %d GGUF parts in %.0fs" % (len(parts), time.time() - t0))
    log("RESULT " + os.path.join(out, "results.json"))


if __name__ == "__main__":
    main()
