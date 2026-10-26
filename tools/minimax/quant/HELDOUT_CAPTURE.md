# A held-out set that is a different document, and the layer packs the A1 sweep needs

The first GPTQ pilot (layer 40, IQ2_XXS, experts 0-1) held out every fifth token of the *same*
document its Hessian came from. That measures interpolation along one text; it cannot say whether
the method generalises. This file is the plan and the commands for the real measurement, plus the
pack format the A1 candidate rebuild uses.

## Where the rows come from

| set | document | where |
|---|---|---|
| Hessian fit (calibration) | `long_prose_austen` — Jane Austen, *Pride and Prejudice*, 17 KB from Gutenberg #1342, 4093 tokens | `_local/ref_out/dumps/long/dump-L40`, `_local/gptq/exp-x-L40-long.npy` |
| score (true held-out) | `long_science_darwin` — Charles Darwin, *On the Origin of Species* (1859), ch. III, ~3000 tokens | not captured yet; `heldout_capture.py` |
| score, second domain | `story_lighthouse` (creative) and, if wanted, `math_probability_urn` (math) | same |
| imatrix corpus (unrelated to both) | `calibration.txt` from `make_calibration.py` (Gutenberg + CPython + transformers source); the text is not kept, the built matrix is | `_local/ref_out/imatrix-minimax-m3.1.gguf` |

The 14 held-out prompts are `_local/ref_out/tools/g3_heldout/prompts.jsonl`. They are hand-written
or public-domain documents that appear nowhere in `calibration.txt` (`make_calibration.py` says so
at the top and picks its files accordingly), and none of them is the Austen excerpt the Hessian came
from. Darwin is the useful one for rows: 3000 tokens at top-4 routing is ~12000 expert hits spread
over ~128 experts, the same order as the calibration document produced (431-440 rows for its
busiest expert).

## The capture

The dumps are made by the fp32 reference (`_local/ref_out/tools/ref/stream_run.py`), not by the
engine: it streams one layer at a time, and layer 40's input needs layers 0-39 anyway. That script
is local tooling under the git-ignored `_local/`, not part of this repository. Two changes to it
are needed for this capture and are in place on the PC:

* `--dump-prompts a,b` dumps layer 40 for several documents in one streaming pass (one pass costs
  the same 41 layers; only the per-layer run time scales with the document count).
* `--release-shards` closes the safetensors mmaps after each layer. The original 60-layer fp32 pass
  left them mapped and peaked at 135 GB of RSS; this keeps the process near one layer's footprint.

`tools/minimax/quant/heldout_capture.py` runs the pass, rebuilds each document's layer input with
`exp_input.py` (the same script that built the calibration `exp-x`), and reports the row count per
expert so the pilot can pick experts that have rows. It takes no lock itself — wrap it:

```powershell
$py = "E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\venv-ref\Scripts\python.exe"
python tools/minimax/pc_launch.py --owner quant --est-ram-gb 60 --max-minutes 45 -- `
  $py tools/minimax/quant/heldout_capture.py `
    --doc long_science_darwin --doc story_lighthouse `
    --model-dir "E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash" `
    --ref "E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\ref_out\tools\ref" `
    --out "E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\gptq\heldout"
```

Output: `_local/gptq/heldout/dumps/dump-L40-<id>/`, `exp-x-L40-<id>.npy`, `capture.json` (rows per
expert, top 8). Estimated 25-35 minutes for two documents at bf16 on the 5900X, 30-60 GB of RAM.
One document is enough for the pilot; the second is a cross-domain check.

## The pilot on true held-out rows

`--exp-eval-x/--exp-eval-dumps` are new: those rows are used **only** for `output_rel` and never
enter a Hessian or a per-expert diagonal. With them the fit side can use the whole calibration
document (`--exp-holdout 0`), and the score is a document the quantizer has never seen.

Pick the two experts with the most held-out rows from `capture.json`, then one pass each (the
`--exp-start N --exp-experts N+1` pair runs a single expert; two passes cost about 8 minutes
together because only the experts' own work repeats):

```powershell
$q = "tools\minimax\quant\build-msvc\strata-mm-quant.exe"
$ck = "E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash"
$imx = "_local\ref_out\imatrix-minimax-m3.1.gguf"
$x = "_local\gptq\exp-x-L40-long.npy"
$d = "_local\ref_out\dumps\long\dump-L40"
$ex = "_local\gptq\heldout\exp-x-L40-long_science_darwin.npy"
$ed = "_local\gptq\heldout\dumps\dump-L40-long_science_darwin"

foreach ($m in "old", "gptq") {
  foreach ($e in 12, 47) {
    & $q --model $ck --recipe MM-IQ2_XS.json --layers 40-40 --imatrix $imx `
      --exp-layer 40 --exp-type IQ2_XXS --exp-x $x --exp-dumps $d --exp-holdout 0 `
      --exp-eval-x $ex --exp-eval-dumps $ed `
      --exp-method $m --exp-experts ($e + 1) --exp-start $e `
      --exp-out "_local\gptq\hold-$m-e$e.json" --threads 20
  }
}
```

Gate/up take the new method only if it wins here; `down` stays on the current method unless it wins
on the same rows. `output_rel` in each report is now cross-document; `output_tokens` says how many
rows the score rests on.

## Full layer 40: the packs, and what A1 needs

A full 128-expert pass is ~5 hours at IQ2_XXS (237 s per expert with three experts in parallel), so
it is written as fragments and merged, a slice per PC turn. The fragments are the exact bytes, so a
split run and one uninterrupted run produce the same GGUF:

```powershell
# slice k: experts START..START+N-1, one turn each
& $q --model $ck --recipe MM-IQ2_XS.json --layers 40-40 --imatrix $imx `
  --exp-layer 40 --exp-type IQ2_XXS --exp-x $x --exp-dumps $d --exp-holdout 0.2 `
  --exp-method gptq --exp-start 0 --exp-experts 16 `
  --exp-pack-frag "_local\gptq\frag-iq2xxs-gptq" --threads 20
# ... --exp-start 16 --exp-experts 32 and so on, all into the same fragment directory

& $q --model $ck --recipe MM-IQ2_XS.json --layers 40-40 --imatrix $imx `
  --pack-merge "_local\gptq\frag-iq2xxs-gptq" --exp-layer 40 --exp-type IQ2_XXS `
  --exp-pack "_local\gptq\L40-iq2xxs-gptq.gguf"
```

### Pack format (what C1 needs)

`--exp-pack` writes a small GGUF, `general.architecture` `minimax-m3`, with exactly three tensors:

| tensor | ne | type |
|---|---|---|
| `blk.<L>.ffn_gate_exps.weight` | 6144 x 3072 x 128 | `--exp-type` (IQ2_XXS, IQ1_S, ...) |
| `blk.<L>.ffn_up_exps.weight` | 6144 x 3072 x 128 | same |
| `blk.<L>.ffn_down_exps.weight` | 3072 x 6144 x 128 | same |

plus the attributes `minimax.quant.method`, `minimax.quant.gptq_prior`, `minimax.quant.gptq_damp`,
`minimax.quant.holdout`. Expert `e` occupies row block `e` (the engine reads it as the `e`-th
expert), so a pack is a drop-in donor for one layer:

```
strata-minimax --model <D3 first shard> --expert-source 40-40=pack.gguf ...
```

`--exp-pack-frag` writes the same bytes as `gate-NNN.bin`, `up-NNN.bin`, `down-NNN.bin` (one raw
row-major tensor per file, `NNN` the expert); `--pack-merge` refuses if any of the 384 files is
missing or short.

For candidate C1 (gate/up from IQ2_XXS, down at D3, all MoE layers) the per-layer procedure is one
`--exp-method gptq --exp-type IQ2_XXS` pass over that layer's dumps, then a merge per layer, then one
engine run per session with `--expert-source` for the layers under test (`gptq_layer_session.py`
takes `--arm name=pack.gguf` and passes `--layer`; for several layers pass several
`--engine-arg=--expert-source` flags). Gate/up at IQ1_S is the same command with
`--exp-type IQ1_S`; `down` is left out of the arm by keeping `down-NNN.bin` from the D3 pack (or by
not overriding down at all).
