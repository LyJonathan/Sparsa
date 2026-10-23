"""Build ``prompts.jsonl``: 8 prompts for the R1/R2 reference.

Two prompts are > 2,500 tokens so MSA actually deselects blocks (> 17 blocks
of 128 keys).  Every text is self-written, public domain, or permissively
licensed; see ``SOURCES.md``.

Usage:
    python make_prompts.py --out prompts.jsonl \
        --gutenberg-dir <dir with pp.txt/alice.txt> --python-lib <stdlib dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chat import apply_chat_template

CODE_PROMPT = r"""Write a Python module that implements a small LRU cache with a fixed capacity.

Requirements:
- class LRUCache(capacity)
- get(key) returns the value or None
- put(key, value) inserts or updates and evicts the least recently used entry
- O(1) for both operations
- include type hints and a docstring explaining the eviction rule
- add three doctest examples that cover eviction, update, and a miss

Return only the code in a single fenced block, then a short note about the
time and space complexity."""

PROSE_PROMPT = """Explain, in plain English and without equations, why a mixture-of-experts
language model can have far more parameters than the number of parameters that
are actually used for any one token. Mention routing, the difference between
total and active parameters, and one reason why this trade-off is attractive
for a computer with a fixed amount of memory bandwidth. Finish with a short
paragraph on the main risk of routing."""

CHINESE_PROMPT = """请用简体中文写一篇大约四百字的短文，主题是“为什么在本地电脑上运行大语言模型很有趣”。

要求：
1. 第一段解释“本地运行”和“云端调用”的区别。
2. 第二段讨论显存、内存带宽和模型量化之间的关系，提到 4 比特权重。
3. 第三段给出一个普通用户也能理解的速度与质量之间的取舍例子。
4. 最后用一句话总结你的观点。

请保持语气自然，不要使用营销腔。"""

MATH_PROMPT = """A bakery makes two kinds of bread. A white loaf uses 500 grams of flour and
30 grams of butter, and sells for 4 dollars. A rye loaf uses 400 grams of flour
and 50 grams of butter, and sells for 5 dollars. On Saturday the bakery has
18 kilograms of flour and 1.8 kilograms of butter, and it can bake at most 50
loaves in total.

How many loaves of each kind should it bake to maximize revenue, assuming it
sells everything it bakes? Show the constraints, the objective, and the
arithmetic that identifies the best integer solution. Then answer what happens
to the optimum if the price of a rye loaf rises to 6 dollars."""

TOOL_SYSTEM = "You are a careful assistant. Use the provided tools when they are needed."

TOOL_MESSAGES = [
    {"role": "system", "content": TOOL_SYSTEM},
    {"role": "user", "content": "What is the weather in Reykjavik right now, and should I pack a coat?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Reykjavik", "units": "celsius"},
                }
            }
        ],
    },
    {
        "role": "tool",
        "content": "{\"city\": \"Reykjavik\", \"temperature_c\": 4, \"wind_kph\": 38, \"conditions\": \"rain\"}",
    },
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]

MULTI_TURN = [
    {"role": "system", "content": "You are a concise, friendly tutor."},
    {"role": "user", "content": "I keep forgetting the difference between a process and a thread."},
    {"role": "assistant", "content": "A process owns its own address space; threads share one inside a process."},
    {"role": "user", "content": "So if one thread crashes the whole process dies?"},
    {"role": "assistant", "content": "Usually yes, if the crash is a fault in shared memory or the runtime. An unhandled exception in one thread can take the process down."},
    {"role": "user", "content": "Then why use threads at all?"},
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def make_prompts(gutenberg_dir: Path, python_lib: Path):
    pp = _read(gutenberg_dir / "pp.txt")
    alice = _read(gutenberg_dir / "alice.txt")
    # Strip the Gutenberg header/footer markers.
    start = pp.find("It is a truth universally acknowledged")
    long_prose_body = pp[start : start + 17000]
    a_start = alice.find("Alice was beginning to get very tired")
    long_code_body = _read(python_lib / "argparse.py") + "\n\n" + _read(python_lib / "pathlib.py")
    long_code_body = long_code_body[:26000]

    items = [
        {
            "id": "code_lru",
            "kind": "code",
            "template": None,
            "text": CODE_PROMPT,
        },
        {
            "id": "prose_moe",
            "kind": "prose",
            "template": None,
            "text": PROSE_PROMPT,
        },
        {
            "id": "chinese_local_llm",
            "kind": "chinese",
            "template": None,
            "text": CHINESE_PROMPT,
        },
        {
            "id": "tool_weather",
            "kind": "tool_call",
            "template": "chat",
            "messages": TOOL_MESSAGES,
            "tools": TOOLS,
        },
        {
            "id": "math_bakery",
            "kind": "math",
            "template": None,
            "text": MATH_PROMPT,
        },
        {
            "id": "multi_turn_threads",
            "kind": "multi_turn",
            "template": "chat",
            "messages": MULTI_TURN,
        },
        {
            "id": "long_prose_austen",
            "kind": "long_prose",
            "template": None,
            "text": long_prose_body
            + "\n\nQuestion: summarize the passage above in five bullet points, then explain in two "
            "sentences what the narrator's attitude towards Mrs Bennet appears to be.",
            "source": "Jane Austen, Pride and Prejudice (1813), public domain, Project Gutenberg ebook 1342",
        },
        {
            "id": "long_code_stdlib",
            "kind": "long_code",
            "template": None,
            "text": long_code_body
            + "\n\nQuestion: what do these two standard-library modules have in common in the way they "
            "parse input, and where do their error-handling strategies differ? Answer in prose.",
            "source": "Python 3.11 standard library argparse.py and pathlib.py, PSF-2.0 license",
        },
    ]

    rendered = []
    for item in items:
        if item.get("template") == "chat":
            text = apply_chat_template(
                item["messages"], tools=item.get("tools"), add_generation_prompt=True
            )
            item["rendered_text"] = text
        else:
            item["rendered_text"] = item.get("text", "")
        rendered.append(item)
    return rendered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gutenberg-dir", default=r"C:\Users\admin\AppData\Local\Temp\opencode"
    )
    ap.add_argument(
        "--python-lib",
        default=r"C:\Users\admin\AppData\Local\Programs\Python\Python311\Lib",
    )
    ap.add_argument("--out", default=str(Path(__file__).with_name("prompts.jsonl")))
    ap.add_argument(
        "--tokenizer",
        default=r"E:\Jon's Projects\AI\Strata Fork Minimax\_ref\minimax\tokenizer.json",
    )
    args = ap.parse_args()

    items = make_prompts(Path(args.gutenberg_dir), Path(args.python_lib))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for item in items:
            n = len(tok.encode(item["rendered_text"], add_special_tokens=False).ids)
            item["n_tokens"] = n
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"{item['id']:24s} {item['kind']:10s} {n:6d} tokens")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
