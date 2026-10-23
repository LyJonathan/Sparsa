"""P2b numerical gate: the runnable synthetic GGUF against tools/minimax/tiny_ref.py.

CPU dense placement (`--minimax-dense-layers-gpu 0`) is the strict check. The GPU placement is reported
with a looser bound (the device GEMV reduces in float32). Q8_0 routed experts are reported, not required
to match the fp32 dequant reference: ggml's vec_dot also quantizes the activation.

    python -m pytest tools/minimax/test_forward.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from make_synthetic_gguf import RUNNABLE, build  # noqa: E402
from tiny_ref import compare, forward, load  # noqa: E402


def _exe():
    candidates = [os.environ.get('STRATA_MINIMAX_EXE'),
                  str(ROOT / 'build' / 'strata-minimax.exe'),
                  str(ROOT / 'build' / 'strata-minimax')]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


EXE = _exe()
pytestmark = pytest.mark.skipif(EXE is None, reason='strata-minimax is not built (run tools/minimax/build_windows.ps1)')


def _ids(n):
    vocab = RUNNABLE['vocab']
    return [(i * 17 + 3) % vocab for i in range(n)]


def _run(gguf, ids, out, dense_gpu, last_k=0):
    tok = out.with_suffix('.txt')
    tok.write_text(' '.join(str(i) for i in ids), encoding='utf-8')
    cmd = [EXE, '--model', str(gguf), '--minimax-allow-nonstandard',
           '--minimax-dense-layers-gpu', str(dense_gpu),
           '--minimax-eval', str(tok), '--minimax-dump-logits', str(out),
           '--max-context', '512', '--cpu-threads', '1']
    if last_k:
        cmd += ['--minimax-last-k', str(last_k)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
    assert p.returncode == 0, p.stderr
    got = np.load(out)
    return got, p.stdout


def _check(reference, got, max_abs, top1, label):
    stats = compare(reference, got)
    print(f'{label}: max_abs={stats["max_abs"]:.3e} kl={stats["kl"]:.3e} top1={stats["top1"]:.3f}')
    assert stats['max_abs'] < max_abs, stats
    assert stats['top1'] >= top1, stats
    assert np.isfinite(got).all()
    return stats


def test_cpu_matches_reference_with_and_without_sparse_selection(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    model = load(gguf)
    short, long = _ids(12), _ids(200)
    # 12 tokens / block 8 = 2 blocks, and top-k is 16, so every block is selected.
    # 200 tokens = 25 blocks, top-k 16, so the indexer actually drops blocks.
    for label, ids in (('cpu-dense-select', short), ('cpu-sparse', long)):
        expect = forward(model, ids)
        got, _ = _run(gguf, ids, tmp_path / f'{label}.npy', dense_gpu=0)
        assert got.shape == expect.shape
        _check(expect, got, max_abs=1e-4, top1=1.0, label=label)


def test_gpu_dense_placement_stays_close_on_the_short_prompt(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    ids = _ids(12)
    expect = forward(load(gguf), ids)
    got, stdout = _run(gguf, ids, tmp_path / 'gpu.npy', dense_gpu=4)
    assert 'LOGITS' in stdout
    _check(expect, got, max_abs=5e-2, top1=0.95, label='gpu-dense-select')


def test_last_k_dumps_the_tail(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    ids = _ids(12)
    got, _ = _run(gguf, ids, tmp_path / 'tail.npy', dense_gpu=0, last_k=4)
    full, _ = _run(gguf, ids, tmp_path / 'full.npy', dense_gpu=0)
    assert got.shape == (4, RUNNABLE['vocab'])
    assert np.allclose(got, full[-4:])


def test_q8_experts_are_finite_and_reported(tmp_path):
    gguf = tmp_path / 'q8.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot', quant_experts='q8_0')
    ids = _ids(12)
    # The reference dequantizes the stored expert bytes and dots them in fp32. The engine's vec_dot also
    # quantizes the activation, so this gap is expected and only has to stay in a sane range.
    expect = forward(load(gguf), ids)
    got, _ = _run(gguf, ids, tmp_path / 'q8.npy', dense_gpu=0)
    _check(expect, got, max_abs=5.0, top1=0.5, label='q8-experts')
