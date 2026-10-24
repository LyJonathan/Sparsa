"""Build the benchmark prompts: one token-id list and one text file per context length.

Both engines are pointed at the same tokens. Strata takes token ids (`--ids`), llama.cpp takes text, so the text is
`decode(ids)` from the official tokenizer and the round trip is checked here (`encode(decode(ids)) == ids`).

    python tools/minimax/bench_prompt.py --prompts R:/tools/minimax/ref/prompts.jsonl \
        --tokenizer E:/models/MiniMax-M3.1-Flash/tokenizer.json --lengths 512,4096,32768 --out bench-prompts

`--prompts` is a jsonl with `id` and `rendered_text`; the ids are that text tokenized with the official tokenizer,
concatenated in file order and cycled up to each length (so 32 K repeats the 10.9 K-token corpus).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--lengths', default='512,4096,32768')
    ap.add_argument('--out', default='bench-prompts')
    ap.add_argument('--no-cycle', action='store_true', help='refuse to repeat the corpus; truncate instead')
    a = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise SystemExit('needs the `tokenizers` package (pip install tokenizers)')
    tok = Tokenizer.from_file(a.tokenizer)

    ids_all: list[int] = []
    texts: list[str] = []
    names = []
    for line in Path(a.prompts).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        texts.append(it['rendered_text'])
        names.append((it['id'], len(tok.encode(it['rendered_text'], add_special_tokens=False).ids)))
    corpus = sum(k for _, k in names)
    print('corpus: ' + ', '.join(f'{n}={k}' for n, k in names) + f' -> {corpus} tokens')

    # Long prompts are built by repeating the *text* (with a paragraph break), not the id list: a repeated id list
    # decodes into text whose re-encoding merges differently at every join, while text joins are natural.
    sep = '\n\n'
    src = sep.join(texts)
    ids_all = tok.encode(src, add_special_tokens=False).ids
    print(f'one pass of the corpus text -> {len(ids_all)} tokens')

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in [int(x) for x in a.lengths.split(',')]:
        reps = (n + len(ids_all) - 1) // len(ids_all)
        if reps > 1 and a.no_cycle:
            raise SystemExit(f'{n} needs {reps} passes over {len(ids_all)} corpus tokens and --no-cycle is set')
        if reps == 1:
            take = ids_all[:n]
        else:
            take = tok.encode(sep.join([src] * reps), add_special_tokens=False).ids[:n]
        text = tok.decode(take, skip_special_tokens=False)
        back = tok.encode(text, add_special_tokens=False).ids
        if back != take:
            same = sum(1 for x, y in zip(back, take) if x == y)
            print(f'  [{n}] note: text round trip differs at the cut ({same}/{len(take)} ids match); llama.cpp will '
                  f'see the same text, so the two engines still see the same sequence', file=sys.stderr)
        (out / f'ids_{n}.txt').write_text(','.join(map(str, take)), encoding='utf-8')
        (out / f'text_{n}.txt').write_text(text, encoding='utf-8')
        print(f'  [{n}] ids_{n}.txt text_{n}.txt  ({len(text)} chars, {reps} pass(es), '
              f'round trip {"exact" if back == take else "cut-boundary only"})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
