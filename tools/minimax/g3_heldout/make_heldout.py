"""Build ``prompts.jsonl``: 8 HELD-OUT prompts for the G3 quality gate.

These are *not* part of the A1-mix selection set.  The six short G3 prompts
(`chinese_local_llm`, `code_lru`, `math_bakery`, `multi_turn_threads`,
`prose_moe`, `tool_weather`) and the two long ones (`long_prose_austen`,
`long_code_stdlib`) are the corpus the mixed-quant layer recipe is chosen on;
this set is the final, independent check, so every domain here is deliberately
different from every domain there.

| id | domain | why it is held out |
|---|---|---|
| `sql_retention_cohort` | analytics SQL (window functions, cohort retention) | G3 code is Python only |
| `science_abstract_critique` | materials-science abstract + methods critique | no scientific paper prose in G3 |
| `story_lighthouse` | creative fiction (flash-fiction opening) | no fiction writing task in G3 |
| `japanese_tea_ceremony` | Japanese | G3's only non-English prompt is Chinese |
| `chem_stoich_road_salt` | chemistry stoichiometry | G3's only quantitative prompt is a linear program |
| `legal_consult_turns` | legal prose, multi-turn consultation (`template: chat`) | legal register absent from G3 |
| `api_pagination_chat` | OpenAPI-spec reading and client design (`template: chat`) | no spec-reading prompt in G3 |
| `long_science_darwin` | ~3,000-token Victorian natural-history prose | G3's long prose is Austen (fiction); this is science |

Format is a copy of ``tools/minimax/ref/make_prompts.py``:

* raw-text prompts carry ``text`` and ``template: null`` and are tokenized
  verbatim - the chat template is **not** applied (the reference does the same
  for 6 of its 8 prompts, so the model sees an untemplated document followed by
  a question);
* chat prompts carry ``messages`` (and optionally ``tools``) with
  ``template: "chat"`` and are rendered through the checkpoint's
  ``chat_template.jinja`` with ``add_generation_prompt=True`` via
  ``tools/minimax/ref/chat.py`` - byte-identical handling to G3;
* ``rendered_text`` is stored for both, so the prompt set is self-contained:
  the reference run only needs the tokenizer, never this script.

Length range is the G3 one (84-5536 tokens): seven prompts of ~150-620 tokens
plus one ~3,000-token document, which is well past 17 blocks of 128 keys and so
actually exercises MSA block selection, like the G3 long prompts.

Sources: everything here is self-written except `long_science_darwin`, which is
Charles Darwin, *On the Origin of Species* (1859), chapter III "Struggle for
Existence" - public domain, Project Gutenberg ebook 1228.  See SOURCES.md.

Usage:
    python make_heldout.py --out prompts.jsonl \
        --gutenberg-dir <dir with pg1228.txt> --tokenizer <tokenizer.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ``chat.py`` lives in the reference tree (branch feat/minimax-ref); only the
# two chat-templated prompts need it, and their output is committed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from chat import apply_chat_template  # noqa: E402

SQL_PROMPT = """You are given one Postgres table:

    events(
        event_id      bigint primary key,
        user_id       uuid not null,
        occurred_at   timestamptz not null,
        event_name    text not null,      -- 'signup', 'activate', 'purchase', 'cancel'
        revenue_cents integer not null    -- 0 for non-revenue events
    )

Write a single query that returns, for every weekly signup cohort (Monday-start
weeks, `date_trunc('week', occurred_at)`), the retention curve of that cohort
for weeks 0 through 8: one row per (cohort_week, week_number) with
`users_in_cohort`, `users_retained` and `retention_pct`.

Requirements:
- a user counts as retained in week N if it has at least one event of any kind
  in the calendar week that is exactly N weeks after its signup cohort week;
- "week 0" retention is therefore the activation rate in the signup week;
- cohorts younger than 8 weeks must still return the weeks they have, with
  `NULL` rather than 0 for weeks that have not happened yet;
- use window functions or a lateral join, not eight separate subqueries;
- make the query index-friendly and say which index you would add.

Then explain, in no more than four sentences, why the naive `LEFT JOIN ...
GROUP BY` version silently over-counts users on days with more than one event."""

SCIENCE_ABSTRACT = """Abstract.

We report a scalable route to epitaxial thin films of the layered van der Waals
ferromagnet Cr2Ge2Te6 on (001)-oriented SrTiO3 by pulsed laser deposition, and
we characterise the coupling between growth kinetics and magnetic anisotropy at
thicknesses from 4 to 24 unit cells. Reflection high-energy electron
diffraction shows layer-by-layer growth only in a narrow window of substrate
temperature (245-275 degrees Celsius) and selenium-to-chromium flux ratio
(2.5-3.1); outside it, films grow in a mixed three-dimensional mode with a
rotational mosaic spread above 0.8 degrees. Superconducting quantum interference
device magnetometry gives a Curie temperature of 63 K at 12 unit cells, within
4 K of the bulk value, and a uniaxial anisotropy constant that falls by a factor
of 2.6 as thickness decreases from 24 to 5 unit cells. X-ray magnetic circular
dichroism confirms that the easy axis remains out of plane down to 6 unit cells
and tilts into the plane at 5 unit cells. We attribute this crossover to
interfacial charge transfer that reduces the crystal-field splitting of the
tellurium p orbitals, supported by density functional theory with an on-site
Hubbard correction. Our results establish a growth window for Cr2Ge2Te6 that is
compatible with back-end-of-line processing and identify a thickness limit for
out-of-plane magnetisation that device engineers must respect.

Author affiliations.

1. Department of Materials Science, University of Groningen, 9747 AG Groningen,
   The Netherlands
2. Zernike Institute for Advanced Materials, 9747 AG Groningen, The Netherlands
3. Diamond Light Source, Harwell Science and Innovation Campus, Didcot OX11 0DE,
   United Kingdom
4. Max Planck Institute for Microstructure Physics, 06120 Halle, Germany

Reviewer request.

Summarise the claim of this abstract in two sentences, then list every sentence
whose quantitative statement is not supported by the description of the
measurement that produced it. For each one, say what control experiment or
statistic would be needed before the statement could be accepted, and rate the
severity as minor, major, or fatal."""

STORY_PROMPT = """Write the opening of a short story, roughly six hundred words, in which a
retired lighthouse keeper discovers that the light has been answering something.
Rules:

- third person, past tense, close on the keeper;
- the supernatural element must never be named or explained, only observed;
- at least one full page of the story is a single continuous action;
- the weather should carry the tension without ever being described as a symbol;
- end the opening on a sentence that could also be read as the last sentence of
  the whole story.

Do not use the words "eerie", "uncanny", "suddenly", or "in that moment".
Write the prose first and then, in three bullets, name the three places where
you deliberately withheld information and what a reader can infer from each."""

JAPANESE_PROMPT = """次の条件で、日本語で文章を書いてください。

テーマ：「家族で受け継いできた、小さな商店の閉店について」

条件：
1. 最初の段落で、店を閉める日の朝の様子を、天気と音を中心に描写すること。
2. 二番目の段落で、三代にわたる商売の歴史を、具体的な品物を三つ以上挙げながら説明すること。
3. 三番目の段落で、常連客との最後の会話を、会話文を交えて描くこと。
4. 最後に、店主がこれからどうするかを一文だけ書くこと。

文体は、静かで落ち着いた普通体（だ・である調）にしてください。
「ありがとう」という言葉は一度だけ使ってください。
漢字とひらがなのバランスは、新聞のコラム程度を目安にしてください。"""

CHEM_PROMPT = """A highway depot keeps 24.0 tonnes of rock salt in a covered shed for winter
road treatment. A sample of the salt is analysed and found to be 91.5 percent
sodium chloride by mass, 6.0 percent calcium sulfate, 1.8 percent insoluble
sand and clay, and 0.7 percent water. The other component is magnesium chloride
hexahydrate.

(a) Calculate the mass of actual sodium chloride in the shed, in tonnes.
(b) Road salt works by freezing-point depression. Assuming complete dissociation
    and ideal dilute behaviour, estimate the freezing point of a saturated brine
    made from this salt at 0 degrees Celsius, given that the solubility of NaCl
    is 359 grams per kilogram of water at that temperature and that the
    cryoscopic constant of water is 1.86 kelvin kilograms per mole. State the
    van 't Hoff factor you use and justify it in one sentence.
(c) The depot spreads 18 grams of salt per square metre of road. How many
    kilometres of a two-lane road, each lane 3.5 metres wide, can be treated
    once with the salt actually in the shed?
(d) The sand and clay fraction does not depress the freezing point but is not
    useless. Give one reason why it is deliberately left in the mix, and one
    environmental cost of leaving it in.

Show every conversion factor explicitly and keep track of units."""

LEGAL_MESSAGES = [
    {
        "role": "system",
        "content": "You are a contracts lawyer reviewing a draft for a client. Be precise, cite the clause "
        "numbers you are discussing, and flag anything that is unenforceable or ambiguous rather than "
        "rewriting it silently.",
    },
    {
        "role": "user",
        "content": "Here is the operative part of a mutual non-disclosure agreement my client has been asked "
        "to sign. Please tell me what is wrong with it.\n\n"
        "1. Definition. \"Confidential Information\" means all information disclosed by either party, in any "
        "form, whether or not marked confidential, including information that is already in the public "
        "domain.\n\n"
        "2. Obligations. The Receiving Party shall use the Confidential Information solely for the Purpose "
        "and shall protect it using at least the same degree of care it uses for its own confidential "
        "information, and in no event less than a reasonable degree of care.\n\n"
        "3. Term. This Agreement expires one year after the date of last disclosure. The obligations of "
        "confidentiality survive for one year after expiry.\n\n"
        "4. Remedies. The Parties agree that any breach will cause irreparable harm for which damages are "
        "an inadequate remedy, and that the Disclosing Party shall be entitled to injunctive relief "
        "without the necessity of posting a bond, in addition to a liquidated sum of 250,000 euros per "
        "breach, which the Parties agree is a genuine pre-estimate of loss.\n\n"
        "5. Residuals. The Receiving Party may use for any purpose the residuals resulting from access to "
        "Confidential Information, where residuals means information in intangible form retained in the "
        "unaided memories of individuals who have had access.\n\n"
        "6. Governing law. This Agreement is governed by the laws of the State of Delaware, and the Parties "
        "submit to the exclusive jurisdiction of the courts of England and Wales.",
    },
    {
        "role": "assistant",
        "content": "Six clauses, and I would not let a client sign this as drafted. The two that are not "
        "curable by negotiation are 5 and 6; the rest are drafting problems.",
    },
    {
        "role": "user",
        "content": "Take them one at a time, in order. For each, name the defect, say who it favours, and give "
        "the minimum redline that fixes it. Then tell me which single clause, if the counterparty refuses "
        "to move on anything, should make my client walk away.",
    },
]

OPENAPI_MESSAGES = [
    {
        "role": "system",
        "content": "You are a backend engineer. Answer from the specification you are given and say so "
        "explicitly when the specification does not determine the answer.",
    },
    {
        "role": "user",
        "content": "This is the relevant part of an OpenAPI 3.1 document for an audit-log service. I have to "
        "write a client that pages through all of it and resumes after a crash.\n\n"
        "paths:\n"
        "  /v1/events:\n"
        "    get:\n"
        "      parameters:\n"
        "        - name: cursor\n"
        "          in: query\n"
        "          schema: {type: string, maxLength: 512}\n"
        "        - name: limit\n"
        "          in: query\n"
        "          schema: {type: integer, minimum: 1, maximum: 500, default: 100}\n"
        "        - name: since\n"
        "          in: query\n"
        "          schema: {type: string, format: date-time}\n"
        "        - name: order\n"
        "          in: query\n"
        "          schema: {type: string, enum: [asc, desc], default: asc}\n"
        "      responses:\n"
        "        '200':\n"
        "          headers:\n"
        "            X-RateLimit-Remaining: {schema: {type: integer}}\n"
        "          content:\n"
        "            application/json:\n"
        "              schema:\n"
        "                type: object\n"
        "                required: [items, next_cursor]\n"
        "                properties:\n"
        "                  items:\n"
        "                    type: array\n"
        "                    items: {$ref: '#/components/schemas/Event'}\n"
        "                  next_cursor:\n"
        "                    type: [string, 'null']\n"
        "        '429':\n"
        "          headers:\n"
        "            Retry-After: {schema: {type: integer}}\n"
        "        '410':\n"
        "          description: cursor expired\n\n"
        "components:\n"
        "  schemas:\n"
        "    Event:\n"
        "      type: object\n"
        "      required: [id, ts, actor, action]\n"
        "      properties:\n"
        "        id: {type: string, format: uuid}\n"
        "        ts: {type: string, format: date-time}\n"
        "        actor: {type: string}\n"
        "        action: {type: string}\n"
        "        payload: {type: object, additionalProperties: true}\n",
    },
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def long_darwin_body(gutenberg_dir: Path) -> str:
    """~14 K characters of chapter III, from a stable sentence anchor."""
    text = _read(gutenberg_dir / "pg1228.txt")
    anchor = "I should premise that I use the term Struggle for Existence"
    start = text.find(anchor)
    if start < 0:
        raise SystemExit(f"anchor not found in {gutenberg_dir / 'pg1228.txt'}")
    body = text[start : start + 14000]
    # cut at the last paragraph break so the sample does not end mid-sentence
    body = body[: body.rfind("\n\n")]
    return body.strip()


def make_prompts(gutenberg_dir: Path):
    items = [
        {"id": "sql_retention_cohort", "kind": "sql", "template": None, "text": SQL_PROMPT},
        {"id": "science_abstract_critique", "kind": "science", "template": None, "text": SCIENCE_ABSTRACT},
        {"id": "story_lighthouse", "kind": "creative", "template": None, "text": STORY_PROMPT},
        {"id": "japanese_tea_ceremony", "kind": "japanese", "template": None, "text": JAPANESE_PROMPT},
        {"id": "chem_stoich_road_salt", "kind": "chemistry", "template": None, "text": CHEM_PROMPT},
        {
            "id": "legal_consult_turns",
            "kind": "legal",
            "template": "chat",
            "messages": LEGAL_MESSAGES,
        },
        {
            "id": "api_pagination_chat",
            "kind": "api_spec",
            "template": "chat",
            "messages": OPENAPI_MESSAGES,
        },
        {
            "id": "long_science_darwin",
            "kind": "long_science",
            "template": None,
            "text": long_darwin_body(gutenberg_dir)
            + "\n\nQuestion: in two paragraphs, reconstruct the argument the author makes about why "
            "the competition between individuals of the same species is the most severe form of "
            "struggle, and identify the two places where the argument relies on an analogy rather "
            "than on an observation. Quote at most four words at a time when you refer to the text.",
            "source": "Charles Darwin, On the Origin of Species (1859), ch. III, public domain, "
            "Project Gutenberg ebook 1228",
        },
    ]

    for item in items:
        if item.get("template") == "chat":
            item["rendered_text"] = apply_chat_template(
                item["messages"], tools=item.get("tools"), add_generation_prompt=True
            )
        else:
            item["rendered_text"] = item.get("text", "")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gutenberg-dir",
        default=str(Path(__file__).resolve().parent),
        help="directory holding pg1228.txt (Origin of Species, Gutenberg ebook 1228)",
    )
    ap.add_argument("--out", default=str(Path(__file__).with_name("prompts.jsonl")))
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    items = make_prompts(Path(args.gutenberg_dir))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    total = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for item in items:
            n = len(tok.encode(item["rendered_text"], add_special_tokens=False).ids)
            item["n_tokens"] = n
            total += n
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"{item['id']:26s} {item['kind']:12s} {n:6d} tokens")
    print(f"{'TOTAL':26s} {'':12s} {total:6d} tokens")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
