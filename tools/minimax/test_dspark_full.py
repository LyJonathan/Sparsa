"""tools/minimax/test_dspark_full.py - the DSpark drafter across a device boundary, on the full model.

The drafter lives on the last card (`devs.back()`, 1 here) and the engine's layer split puts layer 28 at the start of
that card's share, so the taps at 31/45/59 are produced on device 1 while 3/17 are produced on device 0.  Two serve
runs at that placement - plain decode and `--speculative` - must both finish and must emit the same tokens (the greedy
identity DSpark has to keep); the drafter must load with `dev=1` and its tap list intact.

Needs the real full model and a GPU, so it skips unless the model is pointed at:

    STRATA_MINIMAX_FULL_MODEL=D:/hf-d3/IQ2_XS-D3/MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf \
    STRATA_DSPARK_PACK=models/MiniMax-M3.1-Flash/dspark-q4_0.gguf \
    python -m pytest tools/minimax/test_dspark_full.py -q
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

MODEL = os.environ.get("STRATA_MINIMAX_FULL_MODEL", "")
PACK = os.environ.get("STRATA_DSPARK_PACK", str(ROOT / "models" / "MiniMax-M3.1-Flash" / "dspark-q4_0.gguf"))
PROFILE = os.environ.get("STRATA_MINIMAX_PROFILE", "D:/hf-d3/profiles/mm3-d3.bin")
EXE = os.environ.get("STRATA_MINIMAX_EXE", str(ROOT / "build" / "strata-minimax.exe"))
SPLIT = os.environ.get("STRATA_LAYER_SPLIT", "28")
NPROMPT, NGEN = 24, 16


def exe() -> str | None:
    for c in (EXE, str(ROOT / "build" / "strata-minimax.exe"), str(ROOT / "build" / "strata-minimax")):
        if Path(c).exists():
            return c
    return None


pytestmark = [
    pytest.mark.skipif(not MODEL or not Path(MODEL).exists(), reason="STRATA_MINIMAX_FULL_MODEL is not set"),
    pytest.mark.skipif(exe() is None, reason="strata-minimax is not built"),
    pytest.mark.skipif(not Path(PACK).exists(), reason="the DSpark pack is not next to the model"),
]


def ids() -> list[int]:
    src = ROOT / "tools" / "minimax" / "p13_ids.txt"
    if src.exists():
        return [int(x) for x in src.read_text().split()][:NPROMPT]
    return [3 + 17 * i for i in range(NPROMPT)]


def serve(extra: list[str]) -> subprocess.CompletedProcess:
    text = f"GEN {NGEN} temperature=0 seed=1 {' '.join(str(i) for i in ids())}\nSTOP\n"
    cmd = [exe(), "--model", MODEL, "--serve", "--max-context", "16384", "--gpus", "0,1",
           "--layer-split", SPLIT, "--minimax-profile", PROFILE, "--memory-limit-mib",
           os.environ.get("STRATA_MEMORY_LIMIT_MIB", "45000"), "--cpu-threads", "12",
           "--minimax-prefill-batch", os.environ.get("STRATA_PREFILL_BATCH", "4096"),
           "--minimax-prefill-experts", "mmvq", "--minimax-stats"] + extra
    return subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=5400)


def tokens(p: subprocess.CompletedProcess) -> list[str]:
    return [l.split()[1] for l in p.stdout.splitlines() if l.startswith("T ")]


def test_the_drafter_loads_on_the_second_card_and_keeps_the_output_identical():
    plain = serve([])
    assert plain.returncode == 0, plain.stderr[-2500:]
    assert len(tokens(plain)) == NGEN, (plain.stdout[-800:], plain.stderr[-1500:])

    spec = serve(["--dspark-model", PACK, "--speculative", "4"])
    assert spec.returncode == 0, spec.stderr[-2500:]          # no crash through layer 28 and beyond
    assert len(tokens(spec)) == NGEN, (spec.stdout[-800:], spec.stderr[-2500:])
    load = [l for l in spec.stderr.splitlines() if "minimax-dspark pack=" in l]
    assert load, spec.stderr[-2500:]
    dev = int(re.search(r"dev=(\d+)", load[0]).group(1))
    taps = re.search(r"taps=([0-9,]+)", load[0]).group(1)
    assert dev == 1, load[0]                            # the drafter and the 31/45/59 taps are on the second card
    assert taps == "3,17,31,45,59", load[0]
    assert tokens(spec) == tokens(plain), (tokens(plain), tokens(spec))
