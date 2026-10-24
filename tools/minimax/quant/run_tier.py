#!/usr/bin/env python3
"""MM-IQ2_XS / MM-IQ2_XS-D3 job: final llama-gguf-split parts, one command per machine.

Each part is one contiguous layer range, and the part index is fixed so the
files from different machines concatenate in layer order.

PC and rig-2 rates are the measured IQ2_XS expert rate on a Ryzen 9 5900X
(0.053 GB/s logical f32, 8 experts). rig-3 is the measured 0.050 GB/s at 16
threads. One MoE layer is 28.967 GB logical f32 (1.652 TB / 57). Dense layers
are not the long pole.

MM-IQ2_XS-D3 keeps IQ2_XS on ffn_gate_exps/ffn_up_exps and steps the whole
ffn_down_exps tensor up to IQ3_XXS. IQ3_XXS measured faster than IQ2_XS on the
same CPU (0.147 GB/s at 24 threads), so the down tensors add size but little
time. Gate/up and down are projected separately below.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_shards

SPLIT_COUNT = 13

PREFIX_IQ2_XS = "MiniMax-M3.1-Flash-IQ2_XS"
PREFIX_D3 = "MiniMax-M3.1-Flash-IQ2_XS-D3"

# (part, machine, first layer, last layer) inclusive. Part 0 owns KV and globals.
PARTS_IQ2_XS = (
    (0, "rig-2", 0, 4),
    (1, "rig-2", 5, 8),
    (2, "rig-2", 9, 13),
    (3, "rig-2", 14, 14),
    (4, "pc", 15, 25),
    (5, "pc", 26, 35),
    (6, "rig-3", 36, 39),
    (7, "rig-3", 40, 43),
    (8, "rig-3", 44, 47),
    (9, "rig-3", 48, 49),
    (10, "rig-2", 50, 53),
    (11, "rig-2", 54, 58),
    (12, "rig-2", 59, 59),
)

# Balanced for the three machines of the D3 run: 19 / 20 / 18 MoE layers for
# PC / rig-2 / rig-3, about 1.1 h each. rig-2 takes one more than the PC (same
# 24-thread CPU class) and rig-3 takes fewer (16 threads, and it also has to
# stream its shards).
PARTS_D3 = (
    (0, "pc", 0, 3),
    (1, "pc", 4, 7),
    (2, "pc", 8, 11),
    (3, "pc", 12, 15),
    (4, "pc", 16, 21),
    (5, "rig-2", 22, 26),
    (6, "rig-2", 27, 31),
    (7, "rig-2", 32, 36),
    (8, "rig-2", 37, 41),
    (9, "rig-3", 42, 45),
    (10, "rig-3", 46, 50),
    (11, "rig-3", 51, 55),
    (12, "rig-3", 56, 59),
)

PREFIXES = {"MM-IQ2_XS": PREFIX_IQ2_XS, "MM-IQ2_XS-D3": PREFIX_D3}
PARTS_BY_TIER = {"MM-IQ2_XS": PARTS_IQ2_XS, "MM-IQ2_XS-D3": PARTS_D3}

# Logical-f32 GB/s for routed experts, 8-expert IQ2_XS bench, uniform test row.
GBPS = {"pc": 0.053, "rig-2": 0.053, "rig-3": 0.050}
GBPS_NOTE = {
    "pc": "measured, Ryzen 9 5900X, 24 threads",
    "rig-2": "same CPU class as the PC (5900X, 24 threads); not re-benched",
    "rig-3": "measured, Ryzen 7 5700G, 16 threads, 0.604 GB in 12.13 s",
}
# MM-IQ2_XS-D3: (gate/up IQ2_XS GB/s, down IQ3_XXS GB/s), from the real
# one-layer run (--layers 3-3), not from --bench: gate 141.7 s, up 135.2 s,
# down 66.2 s, one whole MoE layer 349.5 s on the PC at 24 threads. bench's
# 0.053 GB/s used a uniform importance row and understated the real rate.
D3_GBPS = {"pc": (0.140, 0.146), "rig-2": (0.140, 0.146), "rig-3": (0.131, 0.140)}
D3_GBPS_NOTE = {
    "pc": "measured on a real one-layer run, 24 threads, gate 141.7 s / up 135.2 s / down 66.2 s",
    "rig-2": "same CPU class as the PC; not re-measured for D3",
    "rig-3": "estimated at 16/24 threads x the PC's real rate (its IQ2_XS bench was 0.94x the PC's)",
}
MOE_F32_GB = 1.652e12 / 57 / 1e9
# Per MoE layer, logical f32: 128 experts x {2 gate/up or 1 down} x 3072 x 6144 x 4.
GATEUP_F32_GB = 128 * 2 * 3072 * 6144 * 4 / 1e9
DOWN_F32_GB = 128 * 3072 * 6144 * 4 / 1e9
# Quantized size projections for the plan table, from the D3 dry run.
GATEUP_Q_GB = 79.612 / 57
DOWN_Q_GB = 52.716 / 57
SIDE_PER_LAYER_GB = (6.818 + 0.251 + 0.179 + 3.429 + 0.003) / 57
DENSE_FFN_PER_LAYER_GB = 0.722 / 3
GLOBALS_GB = 1.306 + 1.008 + 0.003
# Prefetch cap so rig-3 keeps room for the quantized part. rig-2 can hold its set.
PREFETCH_BUDGET_GB = {"pc": 0.0, "rig-2": 220.0, "rig-3": 20.0}


def moe_count(lo, hi):
    return sum(1 for layer in range(lo, hi + 1) if layer >= 3)


def part_name(tier, part):
    return "%s-%05d-of-%05d.gguf" % (PREFIXES[tier], part + 1, SPLIT_COUNT)


def minutes_for(tier, machine, lo, hi):
    moe = moe_count(lo, hi)
    if tier == "MM-IQ2_XS-D3":
        up, down = D3_GBPS[machine]
        return moe * (GATEUP_F32_GB / up + DOWN_F32_GB / down) / 60.0
    return moe * MOE_F32_GB / GBPS[machine] / 60.0


def projected_gb(lo, hi):
    moe = moe_count(lo, hi)
    dense = sum(1 for layer in range(lo, hi + 1) if layer < 3)
    side = (hi - lo + 1) * SIDE_PER_LAYER_GB
    total = moe * (GATEUP_Q_GB + DOWN_Q_GB) + side + dense * DENSE_FFN_PER_LAYER_GB
    if lo == 0:
        total += GLOBALS_GB
    return total


def plan_text(tier):
    lines = []
    if tier == "MM-IQ2_XS-D3":
        lines.append("MM-IQ2_XS-D3. %d final split parts. Experts IQ2_XS, ffn_down_exps IQ3_XXS." % SPLIT_COUNT)
        lines.append("Output tensors about 146.0 GB. Each part is <prefix>-%05d-of-%05d.gguf." % (1, SPLIT_COUNT))
        lines.append("KV metadata is only in part 0. Globals (token_embd, output_norm, output) are only in part 0.")
        lines.append("Do not copy shards or GGUF parts over the PC<->rig LAN (~1 MB/s). Rigs fetch shards from Hugging Face.")
        lines.append("PC uplink to Hugging Face was not measured. Quant wall-clock below is time until every part exists on its machine.")
        lines.append("")
        lines.append("part  %-50s  machine  layers   MoE  GB     minutes" % "file")
        totals = {m: {"moe": 0, "min": 0.0} for m in D3_GBPS}
        for part, machine, lo, hi in PARTS_D3:
            moe = moe_count(lo, hi)
            minutes = minutes_for(tier, machine, lo, hi)
            totals[machine]["moe"] += moe
            totals[machine]["min"] += minutes
            lines.append("%4d  %-50s  %-6s  %2d-%-2d  %3d  %5.1f  %6.1f" % (
                part, part_name(tier, part), machine, lo, hi, moe, projected_gb(lo, hi), minutes))
        lines.append("")
        for machine in ("pc", "rig-2", "rig-3"):
            up, down = D3_GBPS[machine]
            lines.append(
                "%-6s  %2d MoE layers  %5.1f min  (gate/up %.3f GB/s, down %.3f GB/s, %s)"
                % (machine, totals[machine]["moe"], totals[machine]["min"], up, down, D3_GBPS_NOTE[machine])
            )
        makespan = max(totals[m]["min"] for m in totals)
        lines.append("quant wall-clock %.1f min (%.2f h), the slower machine." % (makespan, makespan / 60.0))
        lines.append("Per MoE layer: gate/up %.2f GB f32 + down %.2f GB f32 logical; one layer measured 349.5 s on the PC." % (GATEUP_F32_GB, DOWN_F32_GB))
        lines.append("Dense layers and Q8 attention add a few seconds per layer, not minutes.")
        lines.append("rig-2 has the full checkpoint at /home/adminl/models and hardlinks in ~/strata-work/shards; its 20 layers need ~51 GB of parts.")
        lines.append("rig-3 streams its shards from Hugging Face and deletes a shard after the parts that need it are written.")
        lines.append("PC has the full checkpoint locally and keeps it. It writes 5 parts, ~62 GB.")
        return "\n".join(lines)
    lines.append("MM-IQ2_XS. %d final split parts." % SPLIT_COUNT)
    lines.append("Output tensors about 133.1 GB. Each part is <prefix>-%05d-of-%05d.gguf." % (1, SPLIT_COUNT))
    lines.append("KV metadata is only in part 0. Globals (token_embd, output_norm, output) are only in part 0.")
    lines.append("Do not copy shards or GGUF parts over the PC<->rig LAN (~1 MB/s). Rigs fetch shards from Hugging Face.")
    lines.append("PC uplink to Hugging Face was not measured. Quant wall-clock below is time until every part exists on its machine.")
    lines.append("")
    lines.append("part  file                      machine  layers   MoE  minutes")
    totals = {m: {"moe": 0, "min": 0.0} for m in GBPS}
    for part, machine, lo, hi in PARTS_IQ2_XS:
        moe = moe_count(lo, hi)
        minutes = minutes_for(tier, machine, lo, hi)
        totals[machine]["moe"] += moe
        totals[machine]["min"] += minutes
        lines.append("%4d  %-26s  %-6s  %2d-%-2d  %3d  %6.1f" % (part, part_name(tier, part), machine, lo, hi, moe, minutes))
    lines.append("")
    for machine in ("rig-2", "pc", "rig-3"):
        lines.append(
            "%-6s  %2d MoE layers  %5.1f min  (%.3f GB/s, %s)"
            % (machine, totals[machine]["moe"], totals[machine]["min"], GBPS[machine], GBPS_NOTE[machine])
        )
    makespan = max(totals[m]["min"] for m in totals)
    lines.append("quant wall-clock %.1f min (%.2f h), the slower machine." % (makespan, makespan / 60.0))
    lines.append("Dense layers and Q8 attention add about a minute across the whole model, not per part.")
    lines.append("rig-2 source shards 1-12 and 48 are hardlinked into ~/strata-work/shards and match the HF sizes. The full checkpoint is also under /home/adminl/models. Free space was 39 GB after that download finished, short of the ~50 GB of rig-2 parts held at once. Do not delete /home/adminl/models.")
    lines.append("rig-3 streams 4 shards at a time (~17 GB). The first batch (shards 34-37) is already in ~/strata-work/shards.")
    lines.append("PC layers 15-35 are shards 13-33, already in the local checkpoint. PC does not delete that source.")
    lines.append("Paired shards 4-12 hold layers 5-13 and 50-58. rig-2 keeps them on disk until both ranges are written.")
    lines.append("Layer 14 shares shard 1 with layers 0-2, so part 3 remaps that shard after part 0.")
    return "\n".join(lines)


def machine_parts(tier, machine):
    return [row for row in PARTS_BY_TIER[tier] if row[1] == machine]


def run_quant(args, part, lo, hi):
    out = os.path.join(args.out_dir, part_name(args.tier, part))
    if os.path.isfile(out) and os.path.getsize(out) > 100 * 1024 * 1024 and not args.force:
        print("skip existing", out, flush=True)
        return
    cmd = [
        args.exe,
        "--model", args.model,
        "--recipe", args.tier,
        "--imatrix", args.imatrix,
        "--layers", "%d-%d" % (lo, hi),
        "--split-part", str(part),
        "--split-count", str(SPLIT_COUNT),
        "--out", os.path.join(args.out_dir, PREFIXES[args.tier]),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        subprocess.check_call(cmd)
        return
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run the MM-IQ2_XS / MM-IQ2_XS-D3 split-part job for one machine")
    ap.add_argument("--tier", default="MM-IQ2_XS", choices=sorted(PARTS_BY_TIER))
    ap.add_argument("--machine", choices=("pc", "rig-2", "rig-3"))
    ap.add_argument("--model", default="")
    ap.add_argument("--exe", default="")
    ap.add_argument("--imatrix", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--prefetch", action="store_true", help="download the shards this machine can hold, then stop")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute a part even if its GGUF is already there")
    ap.add_argument("--budget-gb", type=float, default=-1.0)
    args = ap.parse_args()
    if args.plan:
        print(plan_text(args.tier))
        return 0
    if not args.machine:
        print("--machine is required", file=sys.stderr)
        return 2
    rows = machine_parts(args.tier, args.machine)
    if not rows:
        print("no parts for " + args.machine, file=sys.stderr)
        return 2
    if not args.model:
        print("--model is required", file=sys.stderr)
        return 2
    os.makedirs(args.model, exist_ok=True)
    if args.prefetch:
        budget = PREFETCH_BUDGET_GB[args.machine] if args.budget_gb < 0 else args.budget_gb
        if budget <= 0:
            print("prefetch budget is 0 for %s; nothing downloaded" % args.machine, flush=True)
            return 0
        # One fetch across every part, capped, so rig-3 stops after the first batch.
        spec = ",".join("%d-%d" % (lo, hi) for _p, _m, lo, hi in rows)
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "fetch_shards.py"),
            "--dest", args.model,
            "--layers", spec,
            "--sidecars",
            "--budget-gb", str(budget),
        ]
        print(" ".join(cmd), flush=True)
        return subprocess.call(cmd)
    if not args.dry_run and not args.imatrix:
        print(args.tier + " needs --imatrix (a real activation imatrix, not the test one)", file=sys.stderr)
        return 2
    if not args.exe:
        print("--exe is required", file=sys.stderr)
        return 2
    if not args.out_dir:
        print("--out-dir is required", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)
    fetch = os.path.join(os.path.dirname(__file__), "fetch_shards.py")
    for index, (part, _machine, lo, hi) in enumerate(rows):
        spec = "%d-%d" % (lo, hi)
        if args.machine != "pc":
            subprocess.check_call([sys.executable, fetch, "--dest", args.model, "--layers", spec, "--sidecars"])
        elif index == 0:
            subprocess.check_call([sys.executable, fetch, "--dest", args.model, "--sidecars"])
        run_quant(args, part, lo, hi)
        if args.dry_run or args.machine == "pc":
            continue
        keep = []
        for later in rows[index + 1 :]:
            keep.append("%d-%d" % (later[2], later[3]))
        delete_cmd = [sys.executable, fetch, "--dest", args.model, "--layers", spec, "--delete"]
        if keep:
            delete_cmd.extend(["--keep-layers", ",".join(keep)])
        subprocess.check_call(delete_cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
