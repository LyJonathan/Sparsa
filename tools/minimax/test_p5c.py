"""P5c gate: the fused device norm+rope is bit-identical to the host rms_heads/rope/fp16 path.

`head_norm_rope` sums the head's squares sequentially with explicit round-to-nearest intrinsics, tabulates the
SAME double-precision angles the host `rope()` evaluates per call, and pairs the two halves of a NEOX rotation on
one thread, so it does not merely approximate the reference: it reproduces it.  That matters beyond tidiness - a
one-ulp difference in q or k flips an fp16 rounding in the KV cache and the router's 4th expert can follow, which
is how a "numerically fine" path turns into a different answer at token 20.

    python -m pytest tools/minimax/test_p5c.py -q
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


def _exe():
    candidates = [os.environ.get('STRATA_MINIMAX_EXE'),
                  str(ROOT / 'build' / 'strata-minimax.exe'),
                  str(ROOT / 'build' / 'strata-minimax')]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


EXE = _exe()
pytestmark = pytest.mark.skipif(EXE is None, reason='strata-minimax is not built')


def _device_count():
    try:
        import ctypes
        cu = ctypes.WinDLL('nvcuda.dll') if os.name == 'nt' else ctypes.CDLL('libcuda.so.1')
        n = ctypes.c_int(0)
        if cu.cuInit(0) != 0 or cu.cuDeviceGetCount(ctypes.byref(n)) != 0:
            return 0
        return n.value
    except Exception:
        return 0


NDEV = _device_count()


def _ids(n):
    vocab = RUNNABLE['vocab']
    return [(i * 17 + 3) % vocab for i in range(n)]


def _run(gguf, ids, out, extra, max_context='512'):
    tok = out.with_suffix('.txt')
    tok.write_text(' '.join(str(i) for i in ids), encoding='utf-8')
    cmd = [EXE, '--model', str(gguf), '--minimax-allow-nonstandard',
           '--minimax-eval', str(tok), '--minimax-dump-logits', str(out),
           '--max-context', str(max_context), '--cpu-threads', '4', *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode == 0, p.stderr
    return np.load(out), p.stdout, p.stderr


@pytest.fixture(scope='module')
def gguf(tmp_path_factory):
    p = tmp_path_factory.mktemp('p5c') / 'runnable.gguf'
    build(p, geometry=RUNNABLE, weights='glorot')
    return p


def test_token_normrope_is_bit_identical(gguf, tmp_path):
    """Decode path, KV in VRAM, fused device norm+rope against the host loop: 200 tokens, exact."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(200)
    base, _, _ = _run(gguf, ids, tmp_path / 'off.npy',
                      ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1',
                       '--minimax-gpu-normrope', 'off'])
    dev, _, err = _run(gguf, ids, tmp_path / 'on.npy',
                       ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1'])
    assert 'kv_space=vram' in err
    assert np.array_equal(base, dev), float(np.abs(base - dev).max())


def test_batched_normrope_is_bit_identical(gguf, tmp_path):
    """Prefill path: the chunk's fp16 MSA queries and KV rows, with and without the fused kernel."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(200)
    common = ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1',
              '--minimax-prefill-batch', '64']
    base, _, err = _run(gguf, ids, tmp_path / 'boff.npy', common + ['--minimax-gpu-normrope', 'off'])
    assert 'prefill=batch' in err
    dev, _, _ = _run(gguf, ids, tmp_path / 'bon.npy', common)
    assert np.array_equal(base, dev), float(np.abs(base - dev).max())
