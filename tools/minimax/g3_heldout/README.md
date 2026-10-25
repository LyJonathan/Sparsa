# G3 held-out prompt set (`tools/minimax/g3_heldout`)

An independent second gate for the Sparsa MiniMax-M3.1-Flash quantisation work.
The mixed-quant layer recipe ("A1 mix", ~100 GB of experts) is chosen on the
six short G3 prompts (`chinese_local_llm`, `code_lru`, `math_bakery`,
`multi_turn_threads`, `prose_moe`, `tool_weather`, see
`_local/g3_iq2xs/`).  Anything tuned that hard against those prompts has to be
checked on prompts it has never seen, which is what this set is for.

Run artefacts live in `_local/ref_out/heldout/` (git-ignored, this PC) and
`_local/g3_heldout/` (ids + D3 results).  Nothing here is committed as a binary
except the prompt text itself.

## The 8 prompts

| id | domain | tokens | format |
|---|---|---:|---|
| `sql_retention_cohort` | analytics SQL: weekly cohort-retention curve with window functions | 305 | raw text |
| `science_abstract_critique` | materials-science abstract + reviewer-style critique of unsupported claims | 478 | raw text |
| `story_lighthouse` | creative fiction: flash-fiction opening under hard constraints | 164 | raw text |
| `japanese_tea_ceremony` | Japanese: a short structured essay (普通体) | 148 | raw text |
| `chem_stoich_road_salt` | chemistry: stoichiometry, freezing-point depression, unit tracking | 303 | raw text |
| `legal_consult_turns` | legal prose: clause-by-clause review of an NDA, multi-turn | 612 | chat |
| `api_pagination_chat` | OpenAPI 3.1 spec reading + resumable-client design | 577 | chat |
| `long_science_darwin` | long Victorian natural-history prose + an argument-reconstruction question | 3,034 | raw text |
| **total** | | **5,621** | 6 raw / 2 chat |

Two domains are covered twice on purpose, in different registers: scientific
prose appears both as a modern abstract-to-critique task and as a 3,034-token
1859 document, and the two chat prompts exercise the template on a legal and an
API-spec task.

## Why these are held out

Every domain differs from every G3 domain:

| G3 has | held-out has instead |
|---|---|
| Python coding task (`code_lru`) and CPython source (`long_code_stdlib`) | analytics SQL |
| an MoE explainer in plain English (`prose_moe`) | legal opinion register, and a peer-review critique |
| Chinese (`chinese_local_llm`) | Japanese |
| a tool-call conversation (`tool_weather`) | an OpenAPI spec to read, no tools |
| a linear-programming word problem (`math_bakery`) | chemistry with molar masses and units |
| a multi-turn chat about threads (`multi_turn_threads`) | a multi-turn NDA consultation |
| Austen, 1813 fiction (`long_prose_austen`) | Darwin, 1859 scientific prose |

Note that the two `template: chat` prompts still carry the checkpoint's own
system preamble (model version, knowledge cutoff, thinking instructions) because
the template injects it; that is the same code path G3 used, not extra content.

## Format: copied from G3, not reinvented

`make_heldout.py` is a near-copy of `tools/minimax/ref/make_prompts.py`:

* 6 prompts carry `text` with `template: null`.  They are tokenized **verbatim**,
  with the chat template *not* applied - the same handling the reference used for
  6 of its 8 prompts, i.e. the model sees an untemplated document followed by a
  question.
* 2 prompts carry `messages` with `template: "chat"` and are rendered through the
  checkpoint's `chat_template.jinja` with `add_generation_prompt=True`, via
  `tools/minimax/ref/chat.py` (`apply_chat_template`).  Byte-identical to G3.
* `rendered_text` is stored in `prompts.jsonl` for both kinds, so the reference
  run only needs the tokenizer - never this script.  `prompts.jsonl` is
  self-contained and is the artefact of record.
* tokenization is `tokenizer.json` with `add_special_tokens=False`, exactly as
  `stream_run.encode_prompt` does it.
* lengths stay inside the G3 range (84-5,536 tokens).  `long_science_darwin` at
  3,034 tokens is past 17 blocks of 128 keys (2,176), so MSA actually deselects
  blocks - the reason G3 has two >2,500-token prompts.

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
