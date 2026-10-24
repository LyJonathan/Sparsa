"""P9: the automatic VRAM-slot sizing rule (header-only, no GPU, no model).

The 96 GB / 153 GB-model startup OOM came from sizing the slots on `free - reserve` and never subtracting the
batched-prefill workspace that is allocated later.  `include/strata/minimax/vram_plan.hpp` is the fixed rule and
`tests/minimax/vram_plan_test.cpp` asserts its invariant, including the big-card case and the 2x12 GB case.  This test
compiles that small program with the host compiler and runs it; it skips when no MSVC/g++ is installed.

    python -m pytest tools/minimax/test_vram_plan.py -q
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "tests" / "minimax" / "vram_plan_test.cpp"
INC = ROOT / "include"

VCVARS = Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat")


def _build_and_run(tmp: Path) -> str:
    exe = tmp / ("vram_plan_test.exe" if os.name == "nt" else "vram_plan_test")
    if os.name == "nt" and VCVARS.exists():
        cmd = f'call "{VCVARS}" >nul && cl /nologo /EHsc /std:c++17 /O2 /I "{INC}" "{SRC}" /Fe:"{exe}"'
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(tmp))
        if p.returncode != 0:
            return f"compile failed:\n{p.stdout}\n{p.stderr}"
    else:
        cxx = shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            pytest.skip("no MSVC vcvars64.bat and no g++/clang++")
        p = subprocess.run([cxx, "-std=c++17", "-O2", f"-I{INC}", str(SRC), "-o", str(exe)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            return f"compile failed:\n{p.stdout}\n{p.stderr}"
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_vram_slot_plan():
    with tempfile.TemporaryDirectory() as d:
        out = _build_and_run(Path(d))
    assert "ok" in out
