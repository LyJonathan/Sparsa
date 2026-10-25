# G3 held-out prompt set (`tools/minimax/g3_heldout`)

An independent second gate for the Sparsa MiniMax-M3.1-Flash quantisation work.
The mixed-quant layer recipe ("A1 mix", ~100 GB of experts) is chosen on the
six short G3 prompts (`chinese_local_llm`, `code_lru`, `math_bakery`,
`multi_turn_threads`, `prose_moe`, `tool_weather`, see `_local/g3_iq2xs/`).
Anything tuned that hard against those prompts has to be checked on prompts it
has never seen, which is what this set is for.

The set is deliberately larger than a one-off gate needs (14 prompts, not 6-8):
the FP32 reference costs ~30 minutes of A100 time to make and is then reused for
**every future pack**, so breadth is nearly free and re-running it is not.

Run artefacts live in `_local/ref_out/heldout/` (git-ignored, this PC) and
`_local/g3_heldout/` (ids + D3 results).  Nothing here is committed as a binary
except the prompt text itself.

## The 14 prompts

| id | domain | tokens | format |
|---|---|---:|---|
| `sql_retention_cohort` | analytics SQL: weekly cohort-retention curve with window functions | 305 | raw text |
| `shell_log_pipeline` | POSIX `sh` + `awk` streaming log report | 293 | raw text |
| `code_rust_ringbuffer` | Rust `no_std` const-generic ring buffer + tests | 412 | raw text |
| `math_number_theory` | pure maths: characterisation of sums of two squares, with proofs | 384 | raw text |
| `math_probability_urn` | probability: urn sampling and a Polya urn | 277 | raw text |
| `science_abstract_critique` | materials-science abstract + reviewer-style critique | 478 | raw text |
| `story_lighthouse` | creative fiction: flash-fiction opening under hard constraints | 164 | raw text |
| `japanese_tea_ceremony` | Japanese: a short structured essay (普通体) | 148 | raw text |
| `french_atelier_peinture` | French: an essay on "un atelier n'est pas un lieu" | 262 | raw text |
| `chem_stoich_road_salt` | chemistry: stoichiometry, freezing-point depression, units | 303 | raw text |
| `legal_consult_turns` | legal prose: clause-by-clause NDA review, multi-turn | 612 | chat |
| `api_pagination_chat` | OpenAPI 3.1 spec reading + resumable-client design | 577 | chat |
| `multi_turn_debug_session` | multi-turn production-debugging dialogue | 547 | chat |
| `long_science_darwin` | long Victorian natural-history prose + argument reconstruction | 3,034 | raw text |
| **total** | | **7,796** | 11 raw / 3 chat |

Coverage: maths (two kinds), code in three languages (SQL, shell, Rust), two
non-English languages (Japanese, French), legal prose, fiction, a scientific
abstract, a chemistry word problem, three multi-turn/chat prompts, and exactly
one long-context document.  Only the long document is expensive, so the whole
set still costs about 30 minutes of A100 time.

## Why these are held out

Every domain differs from every G3 domain:

| G3 has | held-out has instead |
|---|---|
| Python coding task (`code_lru`) and CPython source (`long_code_stdlib`) | analytics SQL, POSIX shell, and Rust |
| an MoE explainer in plain English (`prose_moe`) | legal opinion register, a peer-review critique, and a debugging dialogue |
| Chinese (`chinese_local_llm`) | Japanese and French |
| a tool-call conversation (`tool_weather`) | an OpenAPI spec to read, no tools |
| a linear-programming word problem (`math_bakery`) | chemistry with molar masses and units; a number-theory proof; an urn/Pólya probability problem |
| a multi-turn chat about threads (`multi_turn_threads`) | a multi-turn NDA consultation and a production incident dialogue |
| Austen, 1813 fiction (`long_prose_austen`) | Darwin, 1859 scientific prose |

Note that the two `template: chat` prompts still carry the checkpoint's own
system preamble (model version, knowledge cutoff, thinking instructions) because
the template injects it; that is the same code path G3 used, not extra content.

## Format: copied from G3, not reinvented

`make_heldout.py` is a near-copy of `tools/minimax/ref/make_prompts.py`:

* 11 prompts carry `text` with `template: null`.  They are tokenized **verbatim**,
  with the chat template *not* applied - the same handling the reference used for
  6 of its 8 prompts, i.e. the model sees an untemplated document followed by a
  question.
* 3 prompts carry `messages` with `template: "chat"` and are rendered through the
  checkpoint's `chat_template.jinja` with `add_generation_prompt=True`, via
  `tools/minimax/ref/chat.py` (`apply_chat_template`).  Byte-identical to G3.
* `rendered_text` is stored in `prompts.jsonl` for both kinds, so the reference
  run only needs the tokenizer - never this script.  `prompts.jsonl` is
  self-contained and is the artefact of record.
* tokenization is `tokenizer.json` with `add_special_tokens=False`, exactly as
  `stream_run.encode_prompt` does it.
* lengths stay inside the G3 range (84-5,536 tokens).  `long_science_darwin` at
  3,034 tokens is past 17 blocks of 128 keys (2,176), so MSA actually deselects
  blocks - the reason G3 has two >2,500-token prompts.  It is the only long
  prompt here, because each one adds ~15 minutes to a reference pass.

## Reproducing

The generator needs the reference tree (`tools/minimax/ref`, branch
`feat/minimax-ref`) on `sys.path`, because it reuses `chat.py`; the reference
tree is not part of this branch.  It also needs `pg1228.txt` (see `SOURCES.md`)
and the checkpoint's `tokenizer.json`.

```sh
# worktree at a checkout of feat/minimax-ref next to this one is the easy path
export MINIMAX_CHAT_TEMPLATE=<checkpoint>/chat_template.jinja
python make_heldout.py --out prompts.jsonl \
    --gutenberg-dir <dir with pg1228.txt> \
    --tokenizer <checkpoint>/tokenizer.json
```

## Reference run (A100, fp32) and scoring

Produced on Thunder Compute, 1x A100-SXM4-80GB, 16 vCPU, 128 GB, torch
2.14.1+cu130, transformers 5.20.2, `stream_run.py --device cuda --dtype fp32`.
All 14 prompts in one pass: **3,295.8 s (55.0 min)** of layer streaming plus
15.0 s for the head, 3.0 s/layer for the six short prompts and ~33 s/layer once
the 3,034-token prompt is in the batch; peak RSS **135.8 GB** (the layer weights
plus mmap'd shards, as the original reference run measured 135.6 GB).

Verification of the run, both done on the box:

* **cross-check against the original G3 reference.** `tool_weather` and
  `code_lru` were re-run through the same code path on the same GPU and compared
  with `_local/ref_out/logits/`: **max abs diff 0.000e+00, exact-match fraction
  100%, KL64 0.0, full-vocab KL 0.0, top-1 100%** over 242 positions. The
  reference is bit-reproducible, so a later re-run needs no explanation budget.
  `compare_xcheck.py` is the script (`_local/ref_out/xcheck/xcheck.json`).
* **self-check of the scorer.** The held-out reference fed back as if it were the
  quant through `g3_sparsa.score_prompt` gives KL64 0.0, full-vocab KL 0.0 and
  top-1 100% on all 1,792 positions (`_local/g3_heldout/selfcheck.json`), so the
  scoring path, the `--minimax-last-k 128` alignment and the ids binaries agree.

## Scoring

The reference is produced by the same code path and precision as the original
G3 reference (`tools/minimax/ref/stream_run.py --device cuda --dtype fp32`), and
saved in the same layout, one directory per prompt:

```
<out>/<id>/token_ids.npy       int32  [T]
<out>/<id>/logits_last128.npy  float32 [min(128,T), 200064]
<out>/<id>/top1.npy            int32  [T]
<out>/<id>/final_hidden.npy    float32 [T, 6144]
<out>/timings.json <out>/meta.json SHA256SUMS
```

```sh
PY stream_run.py --device cuda --dtype fp32 --prompts prompts.jsonl --out <out>
```

Scoring uses the unchanged Strata gate, pointed at the held-out reference and
the held-out ids (the ppl pass is skipped: there is no held-out ppl reference):

```sh
# ids: int32 binaries in the layout g3_sparsa.py expects, one per prompt
for d in <ref>/<id>; do np.fromfile($d/token_ids.npy).astype(np.int32).tofile($ids/<id>.bin); done

python tools/minimax/pc_launch.py --owner heldout -- \
  python tools/minimax/g3_sparsa.py \
    --model 'D:\hf-d3\IQ2_XS-D3\MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf' \
    --out _local/g3_heldout/iq2xs-d3 --tier MM-IQ2_XS-D3 \
    --ref _local/ref_out/heldout --ids _local/g3_heldout/ids \
    --no-ppl --vram-slots 0 --memory-limit-mib 80000 \
    --layer-split auto --cpu-threads 12
```

`--ref` is discovered by `os.listdir`, so it must contain exactly the held-out
prompt directories (plus `meta.json`/`timings.json`, which are files and are
skipped).  `RESULTS.md` in `_local/g3_heldout/` holds the numbers.
