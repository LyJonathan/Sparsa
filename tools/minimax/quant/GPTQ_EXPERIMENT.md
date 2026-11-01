# GPTQ error feedback for the MiniMax IQ experts: one layer, measured

Ported from Project Maya (see `THIRD_PARTY_NOTICES.md`), measured on one MoE layer of
MiniMax-M3.1-Flash with real routed activations, then swapped into the D3 pack through the
engine's `--expert-source`.

## What Maya's recipe is, in their own files

| what | where |
|---|---|
| 256-value blocks, per-block importance `diag(inv(Hinv_SS))`, error pushed through `U = chol(Hinv, upper)` | `tools/maya_quant/gptq.py:70-111` (`gptq()`), `:35` (`BLOCK = 256`) |
| per-expert Hessian `sum w^2 x x^T` + `prior` tokens of the layer-wide Hessian | `gptq.py:170-188` (`round_gate_up`), `:140-147` (`Hbar`) |
| dead channels and `damp * mean(diag)` | `gptq.py:74-79` |
| down from the already-rounded gate/up, and the finding that down measured worse by GPTQ | `gptq.py:191-215`, docstring `:193-194` |
| the baseline it beats (llama.cpp imatrix rounding, per expert) | `gptq.py:218-237` (`rtn_experts`) |
| per-expert imatrix from the calibration pass | `tools/maya_quant/calib.py:82-127` (`StatMoE`), `:177-180` |
| the recipe applied to a full quant, gate/up only | `tools/maya_quant/quantize.py:246-274`, `--gptq*` flags `:151-154` |
| defaults prior 16384, damp 0.1 | `gptq.py:170`, `quantize.py:152-153` |
| held-out split (last 20% of whole sequences) | `gptq.py:285-288` |

Their published Maya-XS number ("error feedback layer by layer on the quantized model itself",
`CHANGELOG.md:140-145`) has no separate source file in the public tree: the DeepSeek run reuses
`gptq.py` with the quantized checkpoint as the source of the statistics. That is the one part of
the recipe this port does **not** reproduce in the measurement below (our Hessians come from the
NVFP4 checkpoint's own forward), and it is the first item of the re-quantization plan.

## Settings

(filled by the run)
