# A1 round 3: the SSD read proof, C1's integrity, and whether C1 fits

This round answers three things the A1 candidate (C1: gate/up from IQ2_XXS, down from D3, in every MoE
layer) could not answer when it was built: whether the quality runs actually read the bytes they claim
to read, whether every one of the 171 expert tensors in the file is its donor's bytes, and whether C1 can
run resident in this box's RAM.

Units: **GB = 1e9 bytes, GiB = 2^30 bytes** (1 GiB = 1.0737 GB). RAM is reported in GiB because Windows
reports it that way.

## 1. The SSD tier is source-aware, and here is the proof it returns the right bytes

The bug: the SSD tier opened its file by **shard index against the base pack**, so an
`--expert-source` layer's bytes came from the base pack's shard of the same index. The fix is two
commits on the engine branch:

* `f99bf4cf` - *the SSD tier is source-aware*.  Every alternate pack's shards are folded into ONE flat
  `ssd_files` list (the base pack's shards, then each `--expert-source` pack's), and `try_must` maps a
  tensor's shard index into that list.  The SSD read, the prefetch and the pin-time fill then open the
  **tensor's own** file with no per-tier knowledge of where it came from.  `--minimax-verify-tensors`
  also reports the resolution through `try_must`, so an alternate tensor prints its own file, shard and
  absolute offset.  Diff: `src/minimax/engine.cpp` (+58/-25), `src/minimax/cuda_ops.{cu,hpp}` (the pinned
  link probe), `tools/minimax/test_expert_source.py` (new), `tools/minimax/verify_pack.py`.
* `c229e53c` - *fix the alternate-source crash*.  The resolution read `alt_shard_base` before the SSD
  block filled it (an out-of-bounds read that faulted right after `decide_norm`).  The bases are now
  computed at open time, where the flat list's order is known, and the SSD block asserts that order.
  Diff: `src/minimax/engine.cpp` (+61/-14), `src/minimax/cuda_ops.*`.

What those commits prove is *resolution*: the loader says it will read the right file.  This round adds
what was missing - proof of what the read path **delivers**:

* the engine gains `--minimax-verify-ssd-hash K`: after an unbuffered read completes, it SHA-256s the
  bytes it just staged, one `INFO minimax-ssd-hash <tensor> <expert> <bytes> <sha256> <shard> <file>`
  line per (tensor, expert), at most K per tensor.  The hashing is opt-in and costs nothing on a normal
  run.  A known-answer line (`sha256("abc")`) is printed first, so a wrong hash cannot make the proof pass.
* `tools/minimax/test_ssd_source_hash.py` forces every expert onto the SSD tier (`--memory-limit-mib 600`,
  no VRAM slots, a real 4-token forward) and runs two arms: the base pack, and the base pack with
  layers 3-7 from the IQ1_M donor.  Every line is compared with the SHA-256 of the same range read
  straight from the pack at the GGUF offset.  It also requires (a) the alt arm to hash experts from
  both packs, (b) the moved block's digests to differ between arms, and (c) a re-hash one expert over
  to differ - three negative controls.

Result: see "Measured this round" below.

**Until that test passes, the block 3-7 (57.44% / 1.2393) and C1 (58.03% / 1.3142) numbers are
UNVERIFIED.**  They were produced by the fixed engine (the binary in this worktree contains both commit
strings), but "produced by the fixed engine" is not a test.

## 2. C1's integrity: every expert tensor, not a sample

`verify_alloc.py` now hashes all 171 expert tensors by default (it sampled 25 before), streaming 64 MiB
at a time and comparing each with the same byte range of the pack the allocation names.  Result: see
"Measured this round".

## 3. Does C1 fit in RAM? No.

Measured on this box, from the engine's own lines and `pc_launch --record-peak`:

| quantity | value | where it comes from |
|---|---:|---|
| physical RAM | 127.9 GiB | GlobalMemoryStatusEx (`128 GiB` nominal) |
| engine host copy of the dense weights | 12.0 GiB | `INFO minimax-mem weights ws=12.0 GiB` (dense on the cards, host copy kept) |
| engine peak working set, all-SSD run | 12.1 GiB | `INFO minimax-mem ... peak=12.1 GiB`, 6-prompt quality runs at `--memory-limit-mib 600` |
| KV at ctx 8192 (f16) | 1.0547 GiB, on the cards | `INFO minimax-kv plan ... kv=1.054688GiB placement=vram` |
| KV at ctx 16384 (f16) | 2.11 GiB, on the cards | doubles with the context |
| non-engine RAM in use at readiness | 25.9 GiB | `avail=89.9 GiB` while the engine held 12.1 GiB |
| C1 experts | 123.72 GB = **115.22 GiB** | the allocation's tensor table |
| C1 whole file | 137.44 GB = **128.00 GiB** | `--minimax-inspect` total |

Fully resident, C1 needs `115.22 GiB` of pinned expert bytes plus the measured `12.1 GiB` of engine
working set above the tier: **127.3 GiB**.  With the 10 GiB floor the box may give the engine at most
`127.9 - 10 = 117.9 GiB` even if nothing else runs.  C1 is **9.4 GiB over**, and with the 25.9 GiB
already in use it is **35.3 GiB over**.  The file alone (128.00 GiB) is the whole physical RAM.

The largest pinned budget that keeps 10 GiB free today is
`127.9 - 25.9 - 10 - 12.1 = 79.9 GiB` (~81,800 MiB).  The decode comparison below uses **60,000 MiB**
(58.6 GiB), which pins 50.9% of C1's experts (56.5 GiB streamed per token) and 47.6% of D3's
(65.6 GiB streamed per token) - the same setting for both, as the comparison requires.  Measured on the
D3 held-out run at that budget: the engine's peak working set is **69.7 GiB** (`--record-peak`), i.e.
11.1 GiB above the pinned tier - so the "127.3 GiB fully resident" figure above is right within 1 GiB,
and the box cannot hold it with 10 GiB to spare.

### Decode-speed plan (not run this round)

`bench_p6_ssd.py` already runs arms in one session and reports `cpu_ram`, `gpu_stream`, `pcie`,
`ssd MB/token` and tok/s.  The command, once the PC is free:

```
python tools/minimax/pc_launch.py --owner maya --est-ram-gb 72 --min-free-gb 10 --record-peak \
  --max-minutes 45 -- python -u tools/minimax/bench_p6_ssd.py \
    --exe build/strata-minimax.exe --label d3 --arm-model D:\hf-d3\...-00001-of-00013.gguf \
    --exe build/strata-minimax.exe --label c1 --arm-model _local/a1/cand_c1.gguf \
    --prompt 249 --decode 256 --rounds 3 --reps 1 --ids <249-token ids> --json-out _local/a1/timing.json \
    --serve --max-context 16384 --gpus 0,1 --layer-split auto --minimax-profile mm3-d3.bin \
    --cpu-threads 12 --minimax-prefill-batch 4096 --minimax-prefill-vram swap \
    --minimax-prefill-experts mmvq --memory-limit-mib 60000 --minimax-profile-time --minimax-stats
```

`--rounds 3` runs D3, C1, D3, C1, D3, C1 - the alternating runs the brief asks for; the median of the
three tok/s per arm is the number.

## 4. The gates

**Held-out set: COMPLETE for both packs (14/14 prompts, 1,792 positions each, 54,000 MiB, one turn
apiece).**

```
held-out   D3   1792 pos   top-1 53.46%   KL64 1.4999
held-out   C1   1792 pos   top-1 53.85%   KL64 1.4138
C1 - D3                   +0.39 pts        -0.0861
```

C1 is **+0.39 points ahead** of D3 with a lower mean KL64 (-0.086).  That is **outside** the 0.3-point
band on the D3-positive side, so the band's "rerun on more tokens" rule does not apply: the held-out
gate is passed on top-1, and the KL agrees.  (The six-prompt set at 679 positions had C1 at -0.29, i.e.
inside the band; the 1,246-position run is the more-tokens answer for that set and is still outstanding.)

**Six-prompt, 1,246 positions: COMPLETE, both packs in one session** (`arm_session`, 600 MiB, the same
tier the 679-position gate used).

```
six-prompt  D3   1246 pos   top-1 56.10%   KL64 1.1186
six-prompt  C1   1246 pos   top-1 54.90%   KL64 1.1245
C1 - D3                    -1.20 pts        +0.0059
```

**C1 trails D3 by 1.20 points on the longer six-prompt set** - outside the 0.3-point band on the negative
side, so the gate's "more tokens" rule has done its job and the answer is negative: the 679-position
-0.29 was not noise, it grows to -1.20 when the earlier positions of each prompt are scored.  The A1 gate
(>= D3 on both sets) therefore **fails on the six-prompt set** even though it passes on the held-out set.

Two attempts at 54,000 and 33,856 MiB were stopped by the launcher's watchdog: WSL held 32-34 GB and a
21.4 GB dirty-page spike took effective RAM to 6.9 GB, under the 10 GB floor.  The completed run is at
600 MiB (all-SSD, peak 12.1 GiB), which is also the exact tier of the 679-position line, so those two
lines are comparable settings as well as budgets (the budget itself is proven logit-neutral).

### Conditioned top-1 (PR #47's slices), per pack

`top1` is against the true next token; `ref` is agreement with the reference's argmax; on-path counts
positions where the reference itself picks the true token.

**Held-out, 1,792 positions:**

| slice | n | D3 top-1 | C1 top-1 | delta |
|---|---:|---:|---:|---:|
| all positions (gate) | 1792 | 53.46% | 53.85% | +0.39 |
| on reference's greedy path | 550 | 78.18% | 78.91% | +0.73 |
| on path, ref p>=0.90 | 308 | **92.21%** | **90.58%** | **-1.63** |
| ref p>=0.99 | 215 | 83.72% | 83.26% | -0.46 |
| ref p>=0.99 (agreement w/ ref) | 215 | 90.70% | 92.09% | +1.39 |

**Six-prompt, 1,246 positions:**

| slice | n | D3 | C1 | delta |
|---|---:|---:|---:|---:|
| all positions (gate) | 1246 | 56.10% | 54.90% | -1.20 |
| on reference's greedy path | 241 | 83.82% | 83.40% | -0.42 |
| on path, ref p>=0.90 | 143 | **96.50%** | **95.10%** | **-1.40** |
| ref p>=0.99 | 138 | 73.91% | 74.64% | +0.73 |
| ref p>=0.99 (agreement w/ ref) | 138 | 91.30% | 92.75% | +1.45 |

**Six-prompt, the earlier 679 positions (re-scored with the same code):**

| slice | n | D3 | C1 | delta |
|---|---:|---:|---:|---:|
| all positions | 679 | 58.32% | 58.03% | -0.29 |
| on reference's greedy path | 188 | 83.51% | 81.38% | -2.13 |
| on path, ref p>=0.90 | 109 | **95.41%** | **93.58%** | **-1.83** |
| ref p>=0.99 | 103 | 73.79% | 73.79% | 0.00 |
| ref p>=0.99 (agreement w/ ref) | 103 | 89.32% | 92.23% | +2.91 |

The pattern is the same on all three sets: C1 is roughly level with D3 on the open-ended positions and on
the reference's greedy path, but **1.4-1.8 points behind on the confident on-path slice** - the slice a
generation harness scores - while it is *better* at reproducing the reference's own confident picks
(ref p>=0.99 agreement +1.4 to +2.9).  So the 8.61 GB C1 saves cost exactly the tokens a generation loop
would emit most confidently; that is the number a "does A1 ship" decision has to be made on.


Turn accounting: 39.4 min wall.  C1's six shorts took 14.0 min, its long prompt 13.3 min (the same
prompt cost D3 18 min), leaving 11.7 min for the six-prompt arm.  Peak working set 64.2 GiB; WSL held
15.1 GB and was not touched.




## 5. Activation-weighted `output_rel`

How the Maya GPTQ pilot gets its expert inputs (`tools/minimax/quant/exp_input.py`):

* The per-op dumps hold `attn_input` (the input-layernorm output), `attn_out` and `router_logits`.
  The expert input is `x = post_attention_layernorm(residual + attn_out)`, and RMSNorm only scales the
  residual, so `residual = attn_input / (w_in + 1)` times **one unknown scalar per token**.
* That scalar is fitted by golden-section search on `log|x @ gate_w^T - router_logits|` (128 equations
  per token), then the reconstruction is checked against the dumped `expert_XXX_gate_up` of every
  expert that has tokens.  On layer 40 it matches to ~1e-6 relative - a real activation file, which is
  why `strata-mm-quant --exp-x` works there.
* `output_rel_probe.py` uses the same reconstruction on a **published pack's stored bytes**
  (`||xW^T - xQ^T|| / ||xW^T||`), with two necessary fixes: layers 0/3/40 are the only dumps, and
  `input_layernorm.weight` of this checkpoint has a channel at exactly -1 (a norm weight of 0 after the
  bake), which is masked rather than approximated.  That was layer 3's earlier NaN-and-1.74 failure;
  masked, the reconstruction is usable.

Rank correlation against the measured block deltas: see "Measured this round".  The honest limit is that
only layer 3 sits inside a measured block (3-7); layer 40 is not in 44-48.

## 6. What the ~100 GB path needs from the GPTQ task

To rebuild C1 with the Maya method on gate/up and then put gate/up at IQ1_S:

* **Pack format**: one ordinary GGUF per layer (or a multi-layer file) whose tensor names are exactly
  `blk.<L>.ffn_{gate,up,down}_exps.weight`, with the same shapes as D3 and experts major (gate = w1,
  up = w3, down = w2), plus whichever scale tensors the chosen type needs.  The engine consumes it with
  `--expert-source <L>-<L>=<pack.gguf>`; no second 130 GB file is written for a sweep.
* **Which layers**: the sweep's measured blocks first (3-7, 20-24, 44-48), then 8-12, 32-36, 53-57 so the
  block ranking is complete; the ~100 GB pack needs all 57 MoE layers.
* **Recipe**: GPTQ error feedback on gate/up with the real activations (`--exp-x`, fitted per token as
  above), Hessians from the same dump, per-expert prior as in the pilot; down either from the rounded
  gate/up or left at D3, measured both ways.
* **Inputs I can supply**: engine-written per-op dumps for any layer (`--minimax-dump-dir` +
  `--minimax-dump-layers N`), in the same layout the quantizer already reads; ask for a layer list and a
  prompt and I will capture them in a PC turn.
* **What I need back**: per layer, `output_rel` on held-out tokens for the new pack and for the pack it
  replaces (so the cost table can be calibrated), and the pack's tensor table (type + bytes per tensor)
  so `verify_alloc.py` can prove the built candidate N/N.

## 7. The sweep

Blocks 8-12, 32-36, 53-57 and the resume of 44-48, all six prompts, once the held-out D3 and C1 work is
done.  `arm_session.py --arms "d3:;b8-12:8-12;b32-36:32-36;b53-57:53-57;b44-48:44-48"` with the IQ1_M
donor, `--memory-limit-mib 600`, one lock turn; `arm_session` resumes each arm's per-prompt dumps.

## Measured this round

### The SSD proof (item 1) - PASS

Built from `f99bf4cf` + `c229e53c` + this round's hash hook (`src/core/sha256.cpp`,
`--minimax-verify-ssd-hash`).  `python tools/minimax/test_ssd_source_hash.py --no-launch`, both arms at
`--memory-limit-mib 600`, `prefill=batch`, 4 tokens:

```
base: 171/171 SSD-tier byte ranges match their source pack (D3 shards 1-13)
alt:  171/171 SSD-tier byte ranges match their source pack (IQ1_M-00001 x6, IQ1_M-00002 x9, D3 x156)
negative controls: 15 moved tensor(s) differ between arms; the next expert's range differs
PASS: 342 SSD-tier reads across 171 tensors, from both the base pack and IQ1_M, hash to their own source pack
```

The 15 moved tensors are layers 3-7 x gate/up/down, and every one of them hashes to the IQ1_M donor's
bytes while the same names in the base arm hash to D3's.  The SHA-256 known-answer line (`abc`) passed in
both arms.  So the block 3-7 and C1 quality numbers are **verified** on the read path a quality run uses
(the batched prompt pass, `reads=direct`).  Limit: the token path's queued/prefetch reads are not
instrumented; they share the same file resolution (`t.shard` into the flat list) but are not covered by
this test.

### C1's expert tensors (item 2) - 171/171

`verify_alloc.py --threads 4`, streaming SHA-256 of the built file against the allocated donors:

```
171/171 expert tensors match the allocation pack (type and length)
byte check: 171/171 expert tensors identical (SHA-256) to the pack they were allocated from
experts 123.72 GB, file 137.44 GB, 1188 tensors, 416s
PROOF PASS
```

Not a sample: all 171 (`57 x 3`), each compared with the same range of the pack the allocation named.

### `output_rel` (item 5)

Activation-weighted output error of each published pack's stored bytes, on the tokens each expert was
actually routed (8 experts per role per layer; `output_rel_probe.py`, the -1 norm channel masked):

| layer | role | D3 | IQ2_XXS | IQ1_M | IQ1_M / D3 |
|---|---|---:|---:|---:|---:|
| 3 | gate | 0.2392 | 0.2746 | 0.3716 | 1.55 |
| 3 | up | 0.1188 | 0.1362 | 0.1857 | 1.56 |
| 3 | down | 0.1347 | 0.2282 | 0.2579 | 1.92 |
| 40 | gate | 0.2169 | 0.2471 | 0.3274 | 1.51 |
| 40 | up | 0.2942 | 0.3386 | 0.4503 | 1.53 |
| 40 | down | 0.1773 | 0.2939 | 0.3349 | 1.89 |

What it agrees with: the pack order `D3 < IQ2_XXS < IQ1_M` in every role at both layers, which is the
order the six-prompt gate measured (58.32% D3, 50.22% all-IQ2_XXS, 57.44% block 3-7 from IQ1_M); and the
finding that **down is the expensive matrix to move** (1.89-1.92x its own error against 1.51-1.56x for
gate/up), which is why C1 keeps down at D3.

What it cannot do: rank *blocks*.  Only layer 3 lies inside a measured block (3-7); layer 40 is not in
44-48, so with the existing dumps a block-level rank correlation is not computable.  Reporting a
correlation over one block would be an invented number.  Capturing dumps for layers 22 and 46
(`--minimax-dump-dir --minimax-dump-layers 22,46`) is cheap and is the next step if block ranking from
`output_rel` is wanted.

### The longer reference (for the gate's "more tokens")

`full_ref_logits.py` rebuilds the whole sequence's reference logits from each prompt's `final_hidden.npy`
(post-final-norm) and the checkpoint's BF16 `lm_head`, and checks the overlap against the dumped
last-128 logits: cos `1.0002-1.0004`, mean |d| `0.0000` on all six prompts.  Wrote
`_local/ref_out/logits_long` with **1,246** scoreable positions (97 + 114 + 151 + 288 + 84 + 512)
instead of 679, so the six-prompt gate can be run on nearly twice the tokens with KL64 still defined.

### PC minutes

* turn 1a: 10 min (build attempt failed on a missing `pwsh`, 171/171 hashes, `output_rel` L40, first
  reference attempt at the wrong norm convention)
* turn 1b: 1 min (long reference, correct convention)
* turn 1c: 1 min (build)
* SSD proof: 1 min
* turn 2: 61 min wall (13:19-14:20), of which 40 min of engine time on the held-out short phase and 19
  min on the interrupted long-prompt phase; the lock was left held-stale by the killed driver and was
  released by hand.  This overran the 45-minute turn rule - recorded as a process failure, not hidden.
* turn 3: **38.5 min wall**, one lock at 54,000 MiB: D3's two missing held-out prompts (all 14 now
  complete), then C1's first 7 held-out prompts; the driver's 39-minute deadline stopped C1 cleanly at
  `math_probability_urn` with its dumps kept.  Peak working set 64.2 GiB, WSL 9.8 GB untouched.
* turn 4: **39.4 min wall**, 54,000 MiB: C1's six remaining held-out prompts (14.0 min) and its long
  prompt (13.3 min) - the held-out set is complete for both packs - then the six-prompt 1,246-position
  arm, which the clock cut after 11.7 min with 3 of 6 D3 dumps done.  Peak 64.2 GiB, WSL 15.1 GB.
* turn 5: **~39 min wall**.  Two six-prompt attempts at 54,000 and 33,856 MiB were stopped by the
  watchdog (WSL 32-34 GB plus a 21.4 GB dirty-page spike -> 6.9 GB effective).  The same session at
  600 MiB (all-SSD) completed both arms: D3 12.6 min, C1 19.4 min.  Peak 12.1 GiB; WSL 34.3 GB.
* held-out D3 scoring of the 12 complete dumps: seconds, no engine.
* conditioned-metric re-scores of the 679 and held-out dumps with PR #47's g3: seconds, no engine.


