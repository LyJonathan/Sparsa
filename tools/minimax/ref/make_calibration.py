"""Build ``calibration.txt`` for the imatrix: ~45-55K tokens of mixed text.

Sources (all redistributable; see SOURCES.md):
  * English prose  -- Lewis Carroll, Alice's Adventures in Wonderland (1865),
                      public domain, Project Gutenberg ebook 11.
  * Code           -- Python standard library modules from CPython (PSF-2.0)
                      and Apache-2.0 transformers source.  Deliberately a
                      different file set from prompts.jsonl.
  * Spanish        -- Miguel de Cervantes, Don Quijote (1605/1615), public
                      domain, Project Gutenberg ebook 2000.
  * Chinese        -- self-written (this file), public-domain classical quotes.
  * Chat           -- self-written dialogues rendered through the checkpoint's
                      chat template.

No text here appears in prompts.jsonl.  Usage:
    python make_calibration.py --out calibration.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chat import apply_chat_template

CHAT_DIALOGUES = [
    [
        {"role": "user", "content": "Can you explain what a hash table does, in two sentences?"},
        {"role": "assistant", "content": "A hash table maps keys to values by hashing each key to a bucket, so lookups take constant time on average. Collisions are resolved by chaining or open addressing."},
        {"role": "user", "content": "What makes a hash function good for that?"},
    ],
    [
        {"role": "system", "content": "You are a patient writing coach."},
        {"role": "user", "content": "My opening paragraph feels flat. How do I make it less generic?"},
        {"role": "assistant", "content": "Start with a concrete detail rather than a summary. Name one object, one sound, or one action, then let the reader infer the mood."},
        {"role": "user", "content": "Give me an example about a train station."},
    ],
    [
        {"role": "user", "content": "What is the difference between latency and throughput?"},
        {"role": "assistant", "content": "Latency is how long one request takes; throughput is how many requests finish per unit of time. A system can have low latency for one user and still collapse under many users."},
        {"role": "user", "content": "Which one matters more for a chat application?"},
    ],
    [
        {"role": "user", "content": "I have 12 apples and give away a third, then buy 5 more. How many do I have?"},
        {"role": "assistant", "content": "A third of 12 is 4, so you have 8 left. Adding 5 gives 13 apples."},
        {"role": "user", "content": "Now suppose I give away a quarter of the original 12 instead. What then?"},
    ],
    [
        {"role": "user", "content": "Translate 'the early bird catches the worm' into Spanish and explain the image."},
        {"role": "assistant", "content": "A literal rendering is 'el pájaro madrugador atrapa el gusano'. The image is about the advantage of acting early, before competitors or circumstances remove the opportunity."},
        {"role": "user", "content": "Is there an equivalent proverb?"},
    ],
    [
        {"role": "user", "content": "Write a regex that matches ISO dates like 2026-10-23 and rejects impossible months."},
        {"role": "assistant", "content": "A practical pattern is ^(\\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$ and then validate the day for the month in code. Regex alone cannot handle leap years cleanly."},
        {"role": "user", "content": "Why not put the month lengths in the pattern?"},
    ],
]

CHINESE_SELF = """
在本地电脑上运行语言模型，最直接的感受是延迟变得可以预测。请求不再需要穿过公共网络，
也不用排队等待别人的服务器。模型就在你自己的硬盘和内存里，输入一句话，几秒钟之后就能看到结果。
当然，这种可预测性是有代价的：你必须自己准备足够的显存、内存和磁盘空间。

量化是让大模型塞进普通电脑的关键技术之一。把权重从十六位浮点数压缩到四位整数，模型的体积会缩小到原来的四分之一左右，
读取权重所需的内存带宽也随之下降。但压缩并不是免费的午餐，低位宽会丢掉一部分细节，尤其是那些很少被激活的专家层。
因此工程上常见的做法是对不同的层使用不同的精度：注意力层和共享专家保留较高的精度，路由专家则用四位甚至更低的精度。

一个普通用户能理解的例子是看视频。你可以选择高清版本，画面清楚但需要更快的网络；也可以选择标清版本，
加载更快、占用更小，但在快速运动的场景里会看到明显的色块。模型量化也是同样的取舍：速度、内存和回答质量三者之间
需要找到一个平衡点。对大多数日常任务来说，稍微低一点的精度几乎察觉不到，但在数学推理和长代码生成上，差距会变得明显。

所以，本地运行模型的乐趣不仅在于“能跑起来”，更在于你可以亲手调整这些参数，观察质量与速度如何此消彼长。

另一个常被忽略的问题是显存和内存的区别。显卡的显存带宽远高于普通内存，但容量通常只有十几 GB；
系统内存容量大，却要慢好几倍。推理时每一步都要把大量权重从存储读进计算单元，因此“权重放在哪里”
往往比“权重有多少”更能决定速度。把一部分专家层留在内存、只把当前需要的专家送进显存，是一种常见的折中方案。

在实际使用中，我建议先从一个稳定的量化版本开始，把提示词、温度和上下文长度都固定下来，
记录每次回答的长度和主观质量。然后只改动一个变量，例如把某个专家层从四位换成八位，再重新测量。
如果质量没有明显变化，就保留更快的版本；如果出现了重复、跑题或者计算错误，就把那一层退回去。
这样的实验不需要昂贵的设备，一台普通的台式机加上一点耐心就够了。

最后要提醒的是，模型的行为不仅取决于权重精度，也取决于采样参数和提示词格式。同样的模型，
换一个聊天模板，答案的风格可能完全不同。因此在比较两个量化版本之前，先确认它们使用的是同一套
模板、同一个分词器和同样的停止符。只有控制住这些变量，测出来的差异才真正属于权重本身。
""".strip()

CLASSICAL = "学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def build(gutenberg: Path, py_lib: Path, transformers_dir: Path, target_tokens=None):
    parts = []

    # 1. English prose: Alice, from the first chapter, skipping the boilerplate.
    alice = _read(gutenberg / "alice.txt")
    a = alice.find("Alice was beginning to get very tired")
    parts.append(("english_alice", alice[a : a + 60000]))

    # 2. Code: CPython stdlib (PSF-2.0) -- different modules from prompts.jsonl.
    code_files = ["asyncio/base_events.py", "json/decoder.py", "dataclasses.py", "typing.py"]
    code = []
    for rel in code_files:
        p = py_lib / rel
        if p.exists():
            code.append(f"# --- {rel} ---\n" + _read(p)[:14000])
    parts.append(("python_stdlib", "\n\n".join(code)))

    # 3. Apache-2.0 transformers source (different from prompts' stdlib files).
    tf_files = ["generation/utils.py", "tokenization_utils_base.py"]
    tf = []
    for rel in tf_files:
        p = transformers_dir / rel
        if p.exists():
            tf.append(f"# --- transformers/{rel} ---\n" + _read(p)[:12000])
    parts.append(("transformers_apache2", "\n\n".join(tf)))

    # 4. Spanish public domain.
    quijote = _read(gutenberg / "quijote.txt")
    q = quijote.find("En un lugar de la Mancha")
    if q < 0:
        q = 0
    parts.append(("spanish_quijote", quijote[q : q + 22000]))

    # 5. Chinese (self-written + public-domain classical quote).
    parts.append(("chinese_self", CLASSICAL + "\n\n" + CHINESE_SELF))

    # 6. Chat-formatted, self-written.
    chat_chunks = []
    for msgs in CHAT_DIALOGUES:
        chat_chunks.append(apply_chat_template(msgs, add_generation_prompt=False))
    parts.append(("chat_self", "\n".join(chat_chunks)))

    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gutenberg", default=r"C:\Users\admin\AppData\Local\Temp\opencode")
    ap.add_argument(
        "--python-lib", default=r"C:\Users\admin\AppData\Local\Programs\Python\Python311\Lib"
    )
    ap.add_argument(
        "--transformers",
        default=r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\venv-ref\Lib\site-packages\transformers",
    )
    ap.add_argument("--tokenizer", default=r"E:\Jon's Projects\AI\Strata Fork Minimax\_ref\minimax\tokenizer.json")
    ap.add_argument("--out", default=str(Path(__file__).with_name("calibration.txt")))
    ap.add_argument("--max-tokens", type=int, default=52000)
    ap.add_argument("--scale", type=float, default=1.0, help="per-section cap = n*scale (before the global budget)")
    ap.add_argument("--truncate-to-tokens", type=int, default=None)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    parts = build(Path(args.gutenberg), Path(args.python_lib), Path(args.transformers))
    budget = args.max_tokens
    out_parts = []
    n_total = 0
    for name, text in parts:
        n = len(tok.encode(text, add_special_tokens=False).ids)
        take = min(n, max(0, budget - n_total), max(1, int(n * args.scale)))
        if take <= 0:
            continue
        if take < n:
            # token-level truncation, then decode back to text
            ids = tok.encode(text, add_special_tokens=False).ids[:take]
            text = tok.decode(ids)
            n = take
        out_parts.append(text)
        n_total += n
        print(f"{name:22s} {n:6d} tokens")
    out = "\n\n".join(out_parts)
    Path(args.out).write_text(out, encoding="utf-8")
    n = len(tok.encode(out, add_special_tokens=False).ids)
    print(f"wrote {args.out}: {n} tokens, {len(out)} chars")


if __name__ == "__main__":
    main()
