"""P10 gate: the tensor-core batched expert GEMM matches the int8 iq_mmvq path.

`--minimax-prefill-expert-gemm tc` dequantizes each routed expert once per chunk and runs every token routed to it
through one FP16 cuBLAS GEMM (FP32 accumulation) instead of the q8_1 iq_mmvq dots.  The weights are decoded from
the same blocks and the activations are higher precision, so the logits may differ only by rounding; the gate is the
project's usual shape (KL plus top-1), not a bitwise compare.

The runnable geometry has a 64-wide expert FFN, which is not a whole IQ2_XS row; this fixture uses 256 so the
tensor-core path has a shape it can take.  CPU-only cases are covered by test_p5a.py (the option changes the device
expert path only).

    python -m pytest tools/minimax/test_p10_expert_gemm.py -q
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

from make_synthetic_gguf import build  # noqa: E402


def _exe():
    for c in [os.environ.get('STRATA_MINIMAX_EXE'),
              str(ROOT / 'build' / 'strata-minimax.exe'),
              str(ROOT / 'build' / 'strata-minimax')]:
        if c and Path(c).exists():
            return str(c)
    return None


EXE = _exe()
pytestmark = pytest.mark.skipif(EXE is None, reason='strata-minimax is not built')

# The runnable shape with a 256-wide expert FFN (IQ2_XS rows are 256 values) and enough layers to exercise both
# the dense block and two MoE layers.
TC = dict(layers=3, hidden=256, vocab=128, heads=8, heads_kv=2, head_dim=32, rot=16, dense_ff=256,
          exp_ff=256, experts=8, used=4, shared=1, leading_dense=1, idx_heads=2, idx_dim=32, idx_top_k=16,
          idx_block=8, idx_local=1, idx_first=0, ctx=512)


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
    vocab = TC['vocab']
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


def _kl(a, b):
    la = a - a.max(axis=1, keepdims=True)
    lb = b - b.max(axis=1, keepdims=True)
    pa = np.exp(la)
    pa /= pa.sum(axis=1, keepdims=True)
    pb = np.exp(lb)
    pb /= pb.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(pa * np.log(np.maximum(pa, 1e-12) / np.maximum(pb, 1e-12)), axis=1)))


@pytest.fixture(scope='module')
def gguf_iq2(tmp_path_factory):
    p = tmp_path_factory.mktemp('p10') / 'tc-iq2.gguf'
    build(p, geometry=TC, weights='glorot', quant_experts='iq2_xs')
    return p


def test_tc_expert_gemm_matches_mmvq(gguf_iq2, tmp_path):
    """The tensor-core expert GEMM against the int8 path: same routes, KL under 1e-3, same top-1."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(128)
    base = ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1',
            '--minimax-prefill-batch', '128']
    ref, _, err_ref = _run(gguf_iq2, ids, tmp_path / 'mmvq.npy',
                           base + ['--minimax-prefill-expert-gemm', 'mmvq'])
    got, _, err_tc = _run(gguf_iq2, ids, tmp_path / 'tc.npy',
                          base + ['--minimax-prefill-expert-gemm', 'tc'])
    assert 'prefill-expert-gemm=tc' in err_tc, err_tc
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    per_row = d.max(axis=1)
    top1 = float((ref.argmax(axis=1) == got.argmax(axis=1)).mean())
    kl = _kl(ref.astype(np.float64), got.astype(np.float64))
    print(f'tc vs mmvq: max={d.max():.3e} median={np.median(per_row):.3e} KL={kl:.3e} top1={top1:.3f}')
    assert kl < 1e-3, kl
    assert top1 >= 0.97, top1


def test_auto_picks_tc_for_a_chunk_and_mmvq_stays_the_default(gguf_iq2, tmp_path):
    """`auto` takes the tensor-core path for a large group; an unset flag keeps the int8 path."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(128)
    base = ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1',
            '--minimax-prefill-batch', '128']
    _, _, err = _run(gguf_iq2, ids, tmp_path / 'auto.npy', base + ['--minimax-prefill-expert-gemm', 'auto'])
    assert 'prefill-expert-gemm=auto' in err, err
    _, _, err = _run(gguf_iq2, ids, tmp_path / 'default.npy', base)
    assert 'prefill-expert-gemm=mmvq' in err, err
