# Sources and licenses — G3 held-out prompt set

## `prompts.jsonl` (8 prompts)

| id | content | source / license |
|---|---|---|
| `sql_retention_cohort` | Postgres weekly cohort-retention query task | self-written, this file |
| `science_abstract_critique` | synthetic materials-science abstract (Cr2Ge2Te6 thin films) + reviewer request | self-written, this file; the numbers are invented for the task and are not a real paper's claims |
| `story_lighthouse` | creative-fiction writing brief | self-written, this file |
| `japanese_tea_ceremony` | Japanese essay brief about a family shop closing | self-written, this file |
| `chem_stoich_road_salt` | rock-salt stoichiometry / freezing-point problem | self-written, this file; constants are textbook values |
| `legal_consult_turns` | fictional mutual-NDA clauses + multi-turn review | self-written, this file; not legal advice, no real party |
| `api_pagination_chat` | fictional OpenAPI 3.1 fragment + resumable-client question | self-written, this file |
| `long_science_darwin` | ~14,000 characters of ch. III "Struggle for Existence" + a question | Charles Darwin, *On the Origin of Species* (1859), **public domain**, Project Gutenberg ebook **1228** |

`prompts.jsonl` stores the rendered text (`rendered_text`) for every prompt as
well as the source message list, so the set is self-contained: the reference run
needs only the checkpoint's `tokenizer.json`.

## Redistribution

Everything in this directory is redistributable: seven prompts are original to
this repository, and the eighth is public-domain text from Project Gutenberg.
No G3 prompt text is reused, so the held-out set shares no tokens with the
selection set — which is the point.

## `pg1228.txt` (not committed)

`make_heldout.py` slices `pg1228.txt` — the plain-text of Gutenberg ebook 1228 —
starting at the sentence "I should premise that I use the term Struggle for
Existence", 14,000 characters, cut back to the last paragraph break.  The file is
~1 MB and is deliberately **not** stored in git; fetch it when regenerating:

```sh
curl -o pg1228.txt https://www.gutenberg.org/cache/epub/1228/pg1228.txt
```

The generated prompt text is committed, so a normal run of the reference never
touches the network.

## Credit

The model under test is MiniMax-M3.1-Flash by MiniMax
(`MiniMaxAI/MiniMax-M3.1-Flash`), used under the terms of its model license;
the chat-template preamble in the two `template: chat` prompts is that
checkpoint's own text, emitted by its `chat_template.jinja`.
