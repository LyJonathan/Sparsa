# G3 on the Strata engine - measured results

The G3 gate scores a quant run against the fp32 reference in
`Strata-MiniMax/_local/ref_out/logits` with the same metrics as the llama.cpp harness
(`_local/g3_iq2xs/tools/compare.py`): KL over the reference's top-64, full-vocabulary KL,
top-1 agreement, NLL against the true next token, and plain-text perplexity on the
reference's own 2,048 ids. `tools/minimax/g3_sparsa.py` produces the logits with
`strata-minimax --minimax-eval`; nothing in the scoring path is re-derived.

Measured on this PC (Ryzen 9 5900X, 2x RTX 5070 12 GB, 128 GB RAM, Windows, NVMe).
Every tier file was checked byte-for-byte against the Hub's blob sha256 before a run
(`tools/minimax/verify_tier.ps1`).

## Alignment and reference self-check (run first)

`tools/minimax/test_g3_sparsa.py` feeds the reference logits back in as the "quant" side:
KL64 and full-vocab KL are exactly 0, top-1 is 100%, and the NLL is the reference's own -
so no off-by-one, wrong slice or row-order bug can hide in the mapping. It also scans
`ids` shifts -2..+2 and finds shift 0 (row *i* predicts `ids[start+i+1]`) best on all 8
prompts, and pins the reference-only numbers:

| prompt | T | dumped rows | ref NLL | ref top-1 vs true | best shift |
|---|---:|---:|---:|---:|---:|
| chinese_local_llm | 97 | 97 | 6.4415 | 20.83% | 0 |
| code_lru | 114 | 114 | 11.5693 | 6.19% | 0 |
| long_code_stdlib | 5536 | 128 | 8.7071 | 16.54% | 0 |
| long_prose_austen | 4093 | 128 | 5.9154 | 18.11% | 0 |
| math_bakery | 151 | 128 | 3.5188 | 37.01% | 0 |
| multi_turn_threads | 288 | 128 | 6.3709 | 27.56% | 0 |
| prose_moe | 84 | 84 | 5.1431 | 24.10% | 0 |
| tool_weather | 547 | 128 | 5.8358 | 46.46% | 0 |

These reference-only numbers are identical to the ones `compare.py` prints for the same
prompts (`_local/g3_iq2xs/g3_results_d3.json`), including code_lru's unflattering
11.57 NLL / 6.19% top-1: that is the fp32 model's own behaviour on an untemplated prompt
(`_local/g3_iq2xs/results_diagnosis.md` 3.7), not a misaligned window. They are a property
of the reference, so they cannot be improved by a better quantizer or a better engine.

## MM-IQ2_XS-D3 versus the fp32 reference

6 of the 8 prompts (679 positions), `--minimax-kv host --minimax-dense-layers-gpu 0
--minimax-prefill-batch 512 --memory-limit-mib 92000 --vram-slots 0`, 16 threads:

| prompt | pos | KL64 | p99 | max | full-vocab KL | top-1 | NLL quant |
|---|---:|---:|---:|---:|---:|---:|---:|
| chinese_local_llm | 97 | 0.6931 | 3.4647 | 3.9063 | 0.8481 | 63.9% | 6.9344 |
| code_lru | 114 | 3.5511 | 12.7576 | 13.3757 | 4.3010 | 33.3% | 11.0639 |
| math_bakery | 128 | 0.7600 | 5.6097 | 6.6365 | 0.9616 | 59.4% | 3.9351 |
| multi_turn_threads | 128 | 0.6693 | 6.1375 | 10.0173 | 0.7049 | 68.8% | 6.2780 |
| prose_moe | 84 | 0.8031 | 3.6309 | 4.3922 | 0.9576 | 52.4% | 5.5321 |
| tool_weather | 128 | 0.7795 | 6.7980 | 9.6390 | 1.0416 | 74.2% | 6.1281 |
| **overall (679 pos)** | | **1.2110** | **9.5184** | **13.3757** | **1.4722** | **59.35%** | **6.6131** |

Same model, same positions, llama.cpp's own quant run (`g3_results_d3.json`, aggregate
recomputed over these 6 prompts because that file's overall block covers 8):

| metric | llama.cpp | Sparsa engine | delta |
|---|---:|---:|---:|
| KL64 mean | 1.3446 | 1.2110 | -0.1337 |
| top-1 agreement | 59.20% | 59.35% | +0.15 pp |
| KL64 p99 (llama.cpp: all 8 prompts) | 11.5321 | 9.5184 | -2.0137 |
| KL64 max (llama.cpp: all 8 prompts) | 16.3213 | 13.3757 | -2.9456 |

Per prompt: Sparsa is better on 3 (chinese -0.04, code_lru -0.70, tool_weather -0.19) and
worse on 3 (math +0.08, multi_turn +0.02, prose_moe +0.06). The two engines disagree with
*each other* by more than either disagrees with the fp32 reference on the flat prompts:

| prompt | KL Sparsa/fp32 | KL llama.cpp/fp32 | KL Sparsa/llama.cpp | top-1 Sparsa/llama.cpp |
|---|---:|---:|---:|---:|
| chinese_local_llm | 0.6931 | 0.7367 | 0.2083 | 74.2% |
| code_lru | 3.5511 | 4.2517 | 2.8944 | 43.9% |
| math_bakery | 0.7600 | 0.6844 | 0.3057 | 77.3% |
| multi_turn_threads | 0.6693 | 0.6483 | 0.0474 | 94.5% |
| prose_moe | 0.8031 | 0.7383 | 0.2082 | 73.8% |
| tool_weather | 0.7795 | 0.9707 | 0.4298 | 82.0% |

So the engine's graph is not the story: both implementations sit at KL ~0.7-1.0 on the
prompts where the fp32 model is reasonably confident, and both fall apart on code_lru,
where the reference itself is at 11.6 nats. The residual is the IQ2_XS expert tier
(0.32 relative weight error; `results_diagnosis.md` 4).

Timing: 2,089 s of engine time for the six prompts (6 model loads; 151-token prompt
213 s, 288-token 418 s, 547-token 821 s), plus 8.5 s of scoring.

## Not measured

* **long_code_stdlib (5,536 tok) and long_prose_austen (4,093 tok)** - opt-in (`--long-prompts`) and
  not scored here. With the KV cache on the host the batched prefill runs `attend_host` once per
  token per layer, which is O(T^2) and single-threaded: long_code_stdlib had 6,600 s of CPU and 100
  minutes of wall on this build without finishing. The device-attention alternative
  (`--minimax-kv vram|ram`) aborts with an access violation when the dense layers are on the CPU
  (`--minimax-dense-layers-gpu 0`, both `vram` and `ram`, reproduced twice), and with
  `--minimax-dense-layers-gpu -1` it exceeds 12 GB of VRAM per card and stalls. That engine bug is
  being fixed separately; these two prompts are re-runnable as soon as it is.
* **Plain-text perplexity (2,048 ids)** - the pass was started and cut after ~2 minutes at
  the rig handover; it needs the same host-attention path (~30-60 min) and is not measured
  for this tier yet. The llama.cpp harness's D3 number is ppl 87.25 against the fp32
  reference's 39.69.
* IQ3_K and IQ4_KS - not run yet.

## Reproducing

```sh
python tools/minimax/test_g3_sparsa.py                  # alignment + reference self-check
python tools/minimax/g3_sparsa.py --model <part1.gguf> --out _local/g3_sparsa/<run> \
    --tier MM-IQ2_XS-D3 --ref <ref>/logits --ids <ref>/ids \
    --memory-limit-mib 92000 --prefill-batch 512 --cpu-threads 16 --max-context 8192 \
    --vram-slots 0 --dense-layers-gpu 0 --kv host --gpus 0,1 --layer-split auto \
    --validate-with _local/g3_iq2xs/g3_results_d3.json
python tools/minimax/g3_summary.py --root _local/g3_sparsa
```

The default prompt set is the 6 short prompts. `--long-prompts` adds
long_code_stdlib and long_prose_austen (about an hour each on the host-KV path), and
`--prompts a,b` runs an explicit set.
