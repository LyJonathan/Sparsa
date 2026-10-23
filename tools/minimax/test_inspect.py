"""P2a loader tests: the synthetic `minimax-m3` GGUF, the inspect table, and the serve skeleton.

Needs a built `strata-minimax.exe` (tools/minimax/build_windows.ps1 writes it to build/); the tests skip without it.
Run with: python -m pytest tools/minimax/test_inspect.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(0, str(HERE))
from make_synthetic_gguf import build  # noqa: E402


def _exe():
    candidates = [os.environ.get('STRATA_MINIMAX_EXE'),
                  str(ROOT / 'build' / 'strata-minimax.exe'),
                  str(ROOT / 'build' / 'strata-minimax'),
                  str(ROOT.parent / 'Strata-MiniMax' / 'build' / 'strata-minimax.exe')]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


EXE = _exe()
pytestmark = pytest.mark.skipif(EXE is None, reason='strata-minimax is not built (run tools/minimax/build_windows.ps1)')


def run(args, stdin=None):
    return subprocess.run([EXE, *args], input=stdin, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=120)


def test_inspect_synthetic(tmp_path):
    gguf = tmp_path / 'synthetic.gguf'
    build(gguf)
    p = run(['--model', str(gguf), '--minimax-inspect'])
    assert p.returncode == 0, p.stderr
    assert 'MiniMax GGUF: 2 layers, embd 256' in p.stdout
    assert 'geometry: NOT MiniMax-M3.1-Flash' in p.stdout
    assert 'token_embd.weight' in p.stdout
    assert 'blk.0.ffn_gate.weight' in p.stdout          # layer 0 is the dense prefix
    assert 'blk.1.ffn_gate_exps.weight' in p.stdout      # layer 1 is MoE
    assert 'blk.0.indexer.q_proj.weight' in p.stdout     # M3.1: indexer on the dense layers too
    assert 'bytes by class:' in p.stdout
    for cls in ('dense', 'experts', 'embd', 'output', 'total'):
        assert cls in p.stdout
    # The class totals are the sum of that class's tensors, and the rows are all present.
    blk = [l for l in p.stdout.splitlines() if ' blk.' in l]
    assert len(blk) == 2 * 8 + 3 + 8 + 4 + 4      # base per layer + dense MLP + routed/shared + indexer x2
    assert sum(1 for l in p.stdout.splitlines()
               if ' token_embd.weight' in l or ' output.weight' in l or ' output_norm.weight' in l) == 3


def test_inspect_rejects_a_missing_tensor(tmp_path):
    gguf = tmp_path / 'missing.gguf'
    build(gguf, drop='blk.1.ffn_gate_exps.weight')
    p = run(['--model', str(gguf), '--minimax-inspect'])
    assert p.returncode == 1
    assert 'missing tensor blk.1.ffn_gate_exps.weight' in p.stderr


def test_inspect_rejects_a_wrong_shape(tmp_path):
    gguf = tmp_path / 'shape.gguf'
    build(gguf, bad_shape=True)
    p = run(['--model', str(gguf), '--minimax-inspect'])
    assert p.returncode == 1
    assert 'blk.1.ffn_gate_exps.weight has shape' in p.stderr
    assert 'expected' in p.stderr


def test_serve_refuses_a_non_m3_1_file_without_the_development_flag(tmp_path):
    gguf = tmp_path / 'synthetic.gguf'
    build(gguf)
    p = run(['--model', str(gguf), '--serve'], stdin='QUIT\n')
    assert p.returncode == 1
    assert "this is not MiniMax-M3.1-Flash's geometry" in p.stderr
    assert 'READY' not in p.stdout


def test_serve_handshake_and_refusal(tmp_path):
    gguf = tmp_path / 'synthetic.gguf'
    build(gguf)
    p = run(['--model', str(gguf), '--serve', '--minimax-allow-nonstandard', '--max-context', '1024',
             '--vram-reserve-mib', '700', '--batch', '1'],
            stdin='GEN 8 temperature=0.7 1,2,3\nGENI 4 embeddings.bin 1,2\nSTOP\nQUIT\n')
    assert p.returncode == 0, p.stderr
    assert 'INFO engine=minimax-m3' in p.stdout
    assert 'READY 1024 stop' in p.stdout
    assert p.stdout.count('ERR minimax engine: forward pass not implemented yet') == 2
    assert 'DONE' not in p.stdout
    # Unknown serve flags are skipped, not fatal.
    assert 'ignoring --vram-reserve-mib 700' in p.stderr
