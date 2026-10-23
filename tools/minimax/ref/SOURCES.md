# Corpus sources and licenses

## `prompts.jsonl` (8 prompts)

| id | content | source / license |
|---|---|---|
| `code_lru` | LRU-cache coding task | self-written |
| `prose_moe` | explanation of MoE active vs total parameters | self-written |
| `chinese_local_llm` | Chinese essay task | self-written |
| `tool_weather` | tool-call conversation through the chat template | self-written |
| `math_bakery` | linear-programming word problem | self-written |
| `multi_turn_threads` | multi-turn chat about processes/threads | self-written |
| `long_prose_austen` | ~17K chars from *Pride and Prejudice* (1813) + question | Jane Austen, public domain; Project Gutenberg ebook **1342** |
| `long_code_stdlib` | ~26K chars of CPython `argparse.py` + `pathlib.py` + question | CPython standard library, **PSF-2.0** |

`prompts.jsonl` stores the rendered text (`rendered_text`) as well as the
source fields. The two long prompts are 4,093 and 5,536 tokens, so they exceed
17 blocks of 128 and actually exercise MSA block selection.

## `calibration.txt` (~41.5K tokens; used by `imatrix.py`)

| section | tokens | source / license |
|---|---|---|
| English prose | 15,103 | Lewis Carroll, *Alice's Adventures in Wonderland* (1865), public domain, Project Gutenberg ebook **11** |
| Python stdlib code | 13,236 | CPython 3.11 `asyncio/base_events.py`, `json/decoder.py`, `dataclasses.py`, `typing.py`, **PSF-2.0** |
| transformers code | 5,438 | HuggingFace `transformers` 5.20.2 `generation/utils.py`, `tokenization_utils_base.py`, **Apache-2.0** |
| Spanish | 5,841 | Miguel de Cervantes, *Don Quijote de la Mancha* (1605/1615), public domain, Project Gutenberg ebook **2000** |
| Chinese | 305 | self-written + a public-domain quote from the *Analects* |
| chat-formatted | 1,567 | self-written dialogues rendered through the checkpoint's `chat_template.jinja` |

No calibration text overlaps `prompts.jsonl`: the prose books, code files and
chat dialogues are all different documents.

If a corpus file cannot be re-downloaded, the committed `prompts.jsonl` and
`calibration.txt` are self-contained; the `make_*.py` scripts are only needed
to regenerate them (they read Gutenberg/CPython/transformers from local paths
given on the command line).
