# Two candidates after C1's failure: D3-UD (quality up) and a RAM-fit mix

C1 (gate/up IQ2_XXS, down D3, every MoE layer; 123.72 GB of experts) failed the A1 gate: -1.20 points of
top-1 on the 1,246-position six-prompt set and -1.4 points on the confident on-path slice on every set
(`A1_ROUND3.md`, section 4).  This note designs the two follow-ups the brief asked for.  No engine runs
were used: every byte count is read from the packs' GGUF headers, and the only network access was the
Hub file listing (no downloads).

Units: **GB = 1e9, GiB = 2^30**.  Expert sizes below are payload sums over the 171 expert tensors
(`57 x gate/up/down`).

## What we actually have

Local packs (D:), header-verified:

| pack | file | experts | gate/up | down |
|---|---:|---:|---|---|
| **D3** (`hf-d3/...IQ2_XS-D3`) | 146.05 GB / 136.0 GiB | 132.33 GB | IQ2_XS 698.35 MB/tensor | IQ3_XXS 924.84 |
| **IQ2_XXS** (`hf-iq2xxs/...`) | 124.54 GB / 116.0 GiB | 110.81 GB | IQ2_XXS 622.85 | IQ2_XS 698.35 |
| **IQ1_M** (`hf-iq1m/...`) | 106.62 GB / 99.3 GiB | 95.75 GB | IQ1_M 528.48 | IQ2_XXS 622.85 |

All three packs share the same non-expert block: **13.72 GB, Q8_0 dense** (attention 6.82, indexer Q8_0
0.25, shared 3.61, dense FFN 0.72, token_embd 1.31, output Q6_K 1.01, router/norms 0.27).

On the Hub (`LyJonathan/MiniMax-M3.1-Flash-GGUF`) and not local: `IQ1_S` 94.79 GB, `IQ2_XS` 133.14,
`IQ2_KT` 123.82, `IQ3_K` 191.24, `IQ4_KS` 233.56, `NVFP4` 246.11.

Not available in any pack of ours: **Q5_K/Q6_K attention/shared/dense-FFN, F32 indexer, Q4_K/K5_K
embeddings and output, IQ4_XS, IQ3_XXS gate/up**.  Those exist only in Unsloth's `UD-*` files
(`UD-Q2_K_XL` 143.91 GB / 134.0 GiB; `UD-IQ2_XXS` 134.95 GB) or must be requantized from the local
checkpoint (`models/MiniMax-M3.1-Flash`, NVFP4+BF16, 48 shards).

Per-layer savings against D3, from the header table:

| change | saves per layer |
|---|---:|
| gate/up IQ2_XS -> IQ2_XXS | 0.1510 GB |
| down IQ3_XXS -> IQ2_XS (the IQ2_XXS pack's down) | 0.2265 GB |
| gate/up IQ2_XS -> IQ1_M | 0.3397 GB |
| down IQ3_XXS -> IQ2_XXS (the IQ1_M pack's down) | 0.3020 GB |
| all three -> the IQ1_M pack's types | 0.6417 GB |

## The RAM fit ceiling is 114.2 GB of experts, not 124

Measured: the engine's peak working set is the pinned tier + **11.5 GiB** (64.2 GiB at 53.9 GiB pinned),
and the 10 GiB floor plus Windows' own use leaves 117.9 GiB for a fully resident run on an otherwise idle
128 GiB box.  So the largest expert tier that fits is `117.9 - 11.5 = 106.4 GiB = 114.2 GB`.  With the
25.9 GiB that other processes held during this round's measurements the ceiling drops to 86.4 GB.

**A 124 GB expert tier does not fit either.**  C1 at 123.72 GB is already 9.5 GB past the idle ceiling.
So of the brief's "~100-124 GB" band only the lower half is real; the candidates below are sized to it.

## (i) D3-UD - D3 with Unsloth's protections, paid for by their non-expert savings

The recipe, from `_ref/research/UNSLOTH_MINIMAX.md` section 2: attention at Q5_K/Q6_K, shared expert at
Q5_K/Q6_K, dense FFN Q5_K/Q6_K, token_embd Q5_K, output Q4_K, **indexer at F32**, down experts on layers
52/58/59 at IQ4_XS, gate/up on layer 59 at IQ3_XXS.

| part | D3 | D3-UD | delta |
|---|---:|---:|---:|
| non-expert (attention 4.48, indexer 0.94, shared 2.37, dense FFN 0.53, embd 0.85, output 0.69, router/norms 0.27) | 13.72 | **10.04** | -3.68 |
| routed gate/up | 79.61 | 80.06 (L59 IQ3_XXS) | +0.45 |
| routed down | 52.72 | 53.79 (L52/58/59 IQ4_XS) | +1.07 |
| **experts** | 132.33 | **133.85** | +1.52 |
| **file** | 146.05 GB / 136.0 GiB | **143.90 GB / 134.0 GiB** | -2.15 |

That tensor table is **exactly the published `UD-Q2_K_XL`** (143.91 GB, 134.0 GiB, experts 133.86): their
non-expert savings pay for their expert protections, and D3's own experts elsewhere are the same types.
So the cheapest form of this candidate is a download, not a build.

* Donor tensors needed: all 60 layers' non-expert tensors + `blk.52/58/59.ffn_down_exps` (IQ4_XS) +
  `blk.59.ffn_{gate,up}_exps` (IQ3_XXS), i.e. 13.72 GB replaced by 10.04 GB plus 5 expert tensors.
* Have them? **No.**  None of our packs has those types.  Two ways:
  1. download `unsloth/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL` (143.91 GB).  Disk now: C 141.2, D 141.2,
     E 142.8 GB free - each volume is **~2.7 GB short** of 143.91 GB, so this needs a deletion first
     (deleting the failed C1, 137.45 GB, makes E: ~280 GB free and costs nothing we still need).
  2. build from the local checkpoint: requantize 13.72 GB of dense/attention/shared/embd/output to
     Q5_K/Q6_K/Q4_K and the indexer to F32, plus 3 IQ4_XS down and 2 IQ3_XXS gate/up layers with
     `strata-mm-quant`; ~14 GB of output, no download, 2-4 h under the lock.  This is the honest
     fallback if the download is not wanted.
* Size: **143.90 GB / 134.0 GiB**, experts **133.85 GB**.  It is a *quality-up* candidate: 2.15 GB smaller
  than D3 as a file but 1.52 GB **bigger** in the pinned expert tier, and 19.7 GB above the fit ceiling.
* Why it is worth measuring: it is the only candidate that spends its savings on the two things the
  sweep showed matter (the indexer and the late down layers) instead of on attention precision.

## (ii) RAM-fit mix - IQ1_M types only on the cheap layers, never in the tail

The brief's literal design (gate/up IQ2_XXS in cheap blocks only) cannot reach the band: 20 cheap layers
save `20 x 0.151 = 3.02 GB`, i.e. **129.3 GB**, still 15.1 GB above the ceiling and 5.6 GB above C1.
So the candidate has to use the IQ1_M pack's types - which is also the 0.6417 GB/layer the sweep's arms
measured - on the cheap blocks, and only there.

Protection rule (from the brief and from Unsloth): **layers 52-59 never move** (the last 8 MoE layers,
which contain their 52/58/59), and the **20-24 block stays D3** (measured -3.68 points from IQ1_M, the
worst measured block).

Sizes, computed exactly from the header table:

| rung | layers at IQ1_M types | experts | file (+13.72 dense) | fits? (ceiling 114.2) |
|---|---|---:|---:|---|
| gate/up IQ2_XXS only, cheap 20 | 0 | 129.31 | 143.0 | no |
| IQ1_M on 3-7 + 44-48 | 10 | 125.91 | 139.6 | no |
| IQ1_M on 3-7 + 8-12 + 44-48 | 15 | 122.70 | 136.4 | no |
| **IQ1_M on 3-51 except 20-24 and 32-36** | **39** | **107.30** | **121.0** | **yes** (idle box) |
| IQ1_M on 3-51 except 20-24 | 44 | 104.09 | 117.8 | yes |
| IQ1_M on all 3-51 | 49 | 100.88 | 114.6 | yes |

**Recommended rung: the 39-layer one at 107.30 GB of experts / 121.0 GB file** (112.8 GiB), peak working
set ~111.4 GiB, i.e. 6.5 GiB under the idle ceiling.  It keeps D3 on 20-24 (the expensive block), on
32-36 (pending the sweep), and on the whole tail 52-59.

* Donor tensors needed: `blk.{3..19,25..31,37..51}.ffn_{gate,up,down}_exps` from the **IQ1_M** pack (117
  tensors), everything else (dense block + 54 expert tensors) from **D3**.  Both local; no download.
* Build: `mix_layers.py --alloc` with a per-(layer, projection) allocation, or `--expert-source` for the
  sweep without writing a file.  A file is 121 GB, so it needs C1 deleted (or a second volume).
* Size: **107.30 GB experts, 121.0 GB file**, and it is a *fit* candidate, not a gate candidate: at
  IQ1_M on most layers its quality is expected to be well below C1 (which itself failed by 1.20).  Its
  purpose is to establish the smallest pack that runs fully resident and what that costs in points.
* Still missing: which layers are cheap.  The pending sweep (blocks 8-12, 32-36, 53-57, plus the
  resume of 44-48) supplies exactly that.  If 32-36 turns out cheap it is added; if 8-12 turns out
  expensive it is dropped and the same budget moves to 25-31 / 37-43.  The allocation is a one-line
  change because `mix_layers --alloc` takes the per-tensor list.

## What each candidate needs, in one line

| candidate | donors | have? | size (experts / file) | build |
|---|---|---|---|---|
| **D3-UD** | UD-Q2_K_XL (non-expert + 5 protected expert tensors) + D3 (166 expert tensors) | no - 143.91 GB download, or requantize from the local checkpoint | 133.85 / 143.90 GB | download (or 2-4 h requant); ~2.7 GB of disk must be freed on one volume |
| **fit, 39-layer IQ1_M** | IQ1_M (117 expert tensors) + D3 (dense + 54 expert tensors) | **yes** | 107.30 / 121.0 GB | `mix_layers --alloc`, local, no download |
