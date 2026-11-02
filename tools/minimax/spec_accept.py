#!/usr/bin/env python3
"""Prompt-lookup (n-gram) draft acceptance, measured on greedy traces.

Input: a JSON list of {"name": str, "prompt": [token ids], "output": [token ids]}, where `output` is the greedy
(temperature 0) continuation of `prompt` from the engine.  Replays prompt-lookup drafting over that trace:

  * the context is the prompt plus the output so far;
  * the drafter takes the last n tokens (n = 3, then 2) and finds their most recent earlier occurrence in the
    context, the match being extended backwards to its longest length L;
  * a draft is proposed only when L >= the gate; it is the up-to-K tokens that followed that occurrence.

Under greedy verification a draft token is accepted only if it equals the token the model emits at that position,
which is `output[i]` by definition of the trace, so acceptance is measured without running the model again.  Each
verify step emits (accepted drafts + 1) tokens: the verify pass also yields the token after the last accepted one.

Output: per trace and per (K, gate): tokens per step (E), the share of steps that drafted, and the acceptance rate
(accepted / drafted tokens).  E is exact for the trace; it is not a speed.
"""
import argparse
import json
import sys

MAX_BACK = 32


def longest_match(ctx, n):
    """Most recent earlier occurrence of ctx[-n:]; returns (start_after_match, backward_length) or (None, 0)."""
    end = len(ctx)
    if end <= n:
        return None, 0
    key = ctx[end - n:end]
    for start in range(end - n - 1, -1, -1):
        if ctx[start:start + n] == key:
            length = n
            while (length < MAX_BACK and start - (length - n) - 1 >= 0 and
                   ctx[start - (length - n) - 1] == ctx[end - length - 1]):
                length += 1
            return start + n, length
    return None, 0


def propose(ctx, K, gate):
    """Draft for the next positions: n = 3 first, then n = 2; gated on the matched length."""
    for n in (3, 2):
        pos, length = longest_match(ctx, n)
        if pos is None:
            continue
        if length < gate:
            return []
        return ctx[pos:pos + K]
    return []


def replay(prompt, output, K, gate):
    steps = drafted_steps = drafted = accepted = emitted = 0
    i = 0
    hist = list(prompt)
    while i < len(output):
        d = propose(hist + output[:i], K, gate)
        steps += 1
        if d:
            drafted_steps += 1
            drafted += len(d)
            a = 0
            while a < len(d) and i + a < len(output) and d[a] == output[i + a]:
                a += 1
            accepted += a
            step_tokens = min(a + 1, len(output) - i)
            emitted += step_tokens
            i += step_tokens
        else:
            emitted += 1
            i += 1
    return {
        "tokens": len(output),
        "steps": steps,
        "tokens_per_step": round(emitted / max(1, steps), 3),
        "draft_step_share": round(drafted_steps / max(1, steps), 3),
        "drafted": drafted,
        "accepted": accepted,
        "acceptance": round(accepted / max(1, drafted), 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="JSON file of greedy traces")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4], help="draft lengths to try")
    ap.add_argument("--gate", type=int, nargs="+", default=[2, 3, 4, 6], help="minimum matched length to draft")
    args = ap.parse_args()
    with open(args.trace, encoding="utf-8") as f:
        traces = json.load(f)
    for tr in traces:
        print("trace %s: prompt %d tokens, output %d tokens" % (tr["name"], len(tr["prompt"]), len(tr["output"])))
        for K in args.k:
            for gate in args.gate:
                r = replay(tr["prompt"], tr["output"], K, gate)
                print("  K=%d gate=%d  tokens/step %.3f  draft steps %.1f%%  acceptance %.1f%%  (drafted %d, accepted %d)"
                      % (K, gate, r["tokens_per_step"], 100 * r["draft_step_share"], 100 * r["acceptance"],
                         r["drafted"], r["accepted"]))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
