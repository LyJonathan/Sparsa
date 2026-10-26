"""`arm_session.py` without an engine: the ranking table, the resume rule, the arm parser.

The driver's job is to make one comparison out of several arms in one PC turn, so
the parts worth a test are the ones that decide *what a number means*:

  * the table's deltas are against the session's first arm (the baseline), and the
    per-GB column is dKL64 over the bytes the arm really saves (0.6417 GB a layer,
    measured from the two packs' tensor tables - not a rounded guess);
  * an arm whose results.json covers only some of the requested prompts is **not**
    done (a half-run arm reported as done would silently compare 211 positions
    against 679);
  * `name:spec` parses into the layer list the engine is asked to override, and a
    file arm (`name:@pack.gguf`) overrides nothing;
  * the sweep blocks partition the 57 MoE layers (3..59) with no overlap and no gap.

    python tools/minimax/test_arm_session.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_session as A  # noqa: E402

PROMPTS = ['chinese_local_llm', 'code_lru', 'math_bakery', 'multi_turn_threads', 'prose_moe', 'tool_weather']
ROW = {'positions': 679, 'top1_agreement': 0.5832, 'kl64_mean': 1.3332}


def arm_dir(root: Path, name: str, row: dict, prompts=PROMPTS) -> Path:
    d = root / name
    d.mkdir()
    (d / 'results.json').write_text(json.dumps({'overall': row, 'per_prompt': [{'prompt': p} for p in prompts]}),
                                    encoding='utf-8')
    return d


def main() -> int:
    fails = []

    # 1. the table: deltas are against the first arm, and dKL64/GB uses the real bytes
    root = Path(tempfile.mkdtemp())
    arm_dir(root, 'd3', ROW)
    arm_dir(root, 'b3-7', dict(ROW, top1_agreement=0.5720, kl64_mean=1.3900))
    text = A.summary(root, [('d3', ''), ('b3-7', '3-7')], set(PROMPTS))
    rows = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    base, blk = rows[0], rows[1]
    if base['dkl64'] != 0.0:
        fails.append(f'the first arm is not the delta origin: {base["dkl64"]}')
    if abs(blk['dkl64'] - 0.0568) > 1e-9:
        fails.append(f'dKL64 is {blk["dkl64"]}, expected 0.0568')
    if abs(blk['gb_saved'] - 5 * A.GB_PER_LAYER) > 0.005:
        fails.append(f'five layers must save {5 * A.GB_PER_LAYER:.4f} GB, got {blk["gb_saved"]}')
    want = round(0.0568 / (5 * A.GB_PER_LAYER), 4)
    if abs(blk['dkl64_per_gb'] - want) > 1e-3:
        fails.append(f'dKL64/GB is {blk["dkl64_per_gb"]}, expected {want}')
    if '+0.0568' not in text or '| 3.21 |' not in text:
        fails.append('the printed table does not carry the delta and the GB column')
    print(text)

    # 2. a partially scored arm is not done
    part = Path(tempfile.mkdtemp())
    arm_dir(part, 'half', ROW, prompts=PROMPTS[:2])
    if A.scored(part / 'half', set(PROMPTS)) is not None:
        fails.append('a 2-prompt arm was accepted as a complete 6-prompt arm')
    if A.scored(part / 'half', set(PROMPTS[:2])) is None:
        fails.append('a 2-prompt arm was not accepted as a complete 2-prompt arm')

    # 3. arm parsing
    if A.parse_arms('d3:; b3-7:3-7 ;cand:@D:/x.gguf') != [('d3', ''), ('b3-7', '3-7'), ('cand', '@D:/x.gguf')]:
        fails.append('parse_arms lost an arm or its spec')
    if A.layers_of('58-59') != [58, 59] or A.layers_of('30') != [30] or A.layers_of('@p.gguf') != []:
        fails.append('layers_of is wrong')

    # 4. the sweep blocks are a partition of the 57 MoE layers
    blocks = ['3-7', '8-12', '13-17', '18-22', '23-27', '28-32', '33-37', '38-42', '43-47', '48-52', '53-57', '58-59']
    seen = [L for b in blocks for L in A.layers_of(b)]
    if seen != list(range(3, 60)):
        fails.append(f'the blocks are not a partition of 3..59: {seen}')

    for f in fails:
        print('FAIL: ' + f)
    print('PASS: deltas, the per-GB column, the resume rule, the parser and the block partition'
          if not fails else f'{len(fails)} failures')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
