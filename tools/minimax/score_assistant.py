"""Score a quant on assistant turns only, from llama-perplexity's KLD_DUMP file (Strata patch).

Non-assistant positions are mostly numerical noise in MiniMax-M3.1 itself (Baekpica measured the original against
itself: KL 0.565 on all positions, 0.003 on assistant turns), so the yardstick is the assistant answer tokens: from
just after `]~b]ai\n</mm:think>` up to the `[e~[` that ends the turn. That end token is not
counted: our teacher answers stop at 256 tokens, and the eval file appends it.

    python score_assistant.py --tok D:/mm-ref/quant/src/tokenizer.json dump1.txt [dump2.txt ...]
"""
import argparse
import math

from tokenizers import Tokenizer


def find(seq, pat, start):
    n = len(pat)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == pat:
            return i
    return -1


def assistant_mask(ids, tok, count_eos=False):
    head = tok.encode("]~b]ai\n</mm:think>", add_special_tokens=False).ids
    eos = tok.token_to_id("[e~[")
    mask = [False] * len(ids)
    i = 0
    while True:
        i = find(ids, head, i)
        if i < 0:
            return mask
        j = i + len(head)
        while j < len(ids) and ids[j] != eos:
            mask[j] = True
            j += 1
        if j < len(ids) and count_eos:
            mask[j] = True          # the turn's end token (only when the teacher actually generated it)
        i = j


def score(path, tok):
    ids, rows = {}, []
    for line in open(path, encoding="utf8"):
        p = line.split()
        if p[0] == "T":
            ids[int(p[1])] = int(p[2])
        elif p[0] == "K":
            rows.append((int(p[1]), float(p[3]), int(p[4])))
    seq = [ids[i] for i in range(len(ids))]
    mask = assistant_mask(seq, tok)
    sel = [(k, s) for idx, k, s in rows if idx + 1 < len(mask) and mask[idx + 1]]
    n = len(sel)
    if not n:
        return None
    kl = [k for k, _ in sel]
    mean = sum(kl) / n
    sd = math.sqrt(sum((k - mean) ** 2 for k in kl) / max(1, n - 1) / n)
    top = sum(s for _, s in sel) / n
    kl.sort()
    return {"n": n, "kl": mean, "kl_se": sd, "kl_median": kl[n // 2], "kl_p99": kl[min(n - 1, int(n * 0.99))],
            "top1": top, "top1_se": math.sqrt(top * (1 - top) / n), "all_n": len(rows),
            "all_kl": sum(r[1] for r in rows) / len(rows), "all_top1": sum(r[2] for r in rows) / len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("dumps", nargs="+")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tok)
    print(f"{'dump':40s} {'asst n':>7s} {'KL':>14s} {'median':>7s} {'p99':>6s} {'top-1':>15s}   | all-pos KL / top-1")
    for d in a.dumps:
        r = score(d, tok)
        if r is None:
            print(f"{d:40s} no assistant positions")
            continue
        print(f"{d[-40:]:40s} {r['n']:7d} {r['kl']:7.4f} ± {r['kl_se']:.4f} {r['kl_median']:7.4f} {r['kl_p99']:6.2f} "
              f"{100*r['top1']:6.2f}% ± {100*r['top1_se']:.2f}   | {r['all_kl']:.3f} / {100*r['all_top1']:.1f}%")


if __name__ == "__main__":
    main()
