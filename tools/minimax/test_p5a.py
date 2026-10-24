"""P5a gate: batched prompt processing keeps the per-token math, and the decode prefetch changes nothing.

The batched path is per layer: dense projections as one batched GEMM (cuBLAS), MSA attention over the whole chunk
with the top-k block selection done once per query, then the routed experts grouped per expert so each expert's
bytes are read once per chunk instead of once per token.  Every case below compares it against the P4 token path on
the same file and placement.

    python -m pytest tools/minimax/test_p5a.py -q

The CPU-only cases are the exact gate (same dot products, only the loop order changes).  The GPU cases use the
project's usual tolerance shape: a few rows may flip the router's 4th expert (P4 measured the same thing on the real
checkpoint), so the gate is KL plus top-1 agreement plus the per-row median, not only the max.
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


def _kl(a, b):
    la = a - a.max(axis=1, keepdims=True)
    lb = b - b.max(axis=1, keepdims=True)
    pa = np.exp(la)
    pa /= pa.sum(axis=1, keepdims=True)
    pb = np.exp(lb)
    pb /= pb.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(pa * np.log(np.maximum(pa, 1e-12) / np.maximum(pb, 1e-12)), axis=1)))


def _report(a, b, label):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    per_row = d.max(axis=1)
    top1 = float((a.argmax(axis=1) == b.argmax(axis=1)).mean())
    kl = _kl(a.astype(np.float64), b.astype(np.float64))
    print(f'{label}: max={d.max():.3e} median={np.median(per_row):.3e} KL={kl:.3e} top1={top1:.3f}')
    return d, per_row, top1, kl


@pytest.fixture(scope='module')
def gguf(tmp_path_factory):
    p = tmp_path_factory.mktemp('p5a') / 'runnable.gguf'
    build(p, geometry=RUNNABLE, weights='glorot')
    return p


@pytest.fixture(scope='module')
def gguf_iq2(tmp_path_factory):
    p = tmp_path_factory.mktemp('p5a') / 'runnable-iq2.gguf'
    build(p, geometry=RUNNABLE, weights='glorot', quant_experts='iq2_xs')
    return p


def test_batched_cpu_is_the_token_path(gguf, tmp_path):
    """CPU placement: identical dot products, and the chunked listener keeps the causal order."""
    ids = _ids(200)      # 25 blocks of 8: the indexer drops blocks, so MSA is exercised
    tok, _, _ = _run(gguf, ids, tmp_path / 'tok.npy', ['--minimax-dense-layers-gpu', '0'])
    bat, _, err = _run(gguf, ids, tmp_path / 'bat.npy',
                       ['--minimax-dense-layers-gpu', '0', '--minimax-prefill-batch', '64'])
    assert 'prefill=batch' in err
    d, _, top1, _ = _report(tok, bat, 'cpu batch vs token')
    assert d.max() < 1e-4, d.max()
    assert top1 == 1.0


def test_batched_gpu_dense_and_attention_match(gguf, tmp_path):
    """The GPU path: cuBLAS fp32 projections + the batched MSA kernel; only rounding differs."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(200)
    tok, _, _ = _run(gguf, ids, tmp_path / 'tok.npy',
                     ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1'])
    bat, _, _ = _run(gguf, ids, tmp_path / 'bat.npy',
                     ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1',
                      '--minimax-prefill-batch', '64', '--minimax-prefill-experts', 'cpu'])
    d, per_row, top1, kl = _report(tok, bat, 'gpu attention/dense, cpu experts')
    assert np.median(per_row) < 2e-3, np.median(per_row)
    assert kl < 5e-5, kl
    assert top1 >= 0.98, top1


def test_batched_gpu_experts_match(gguf_iq2, tmp_path):
    """The grouped int8 expert GEMM over a chunk (IQ2_XS gate/up, F32 down) against the token path."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(200)
    tok, _, _ = _run(gguf_iq2, ids, tmp_path / 'tok.npy',
                     ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1'])
    bat, _, _ = _run(gguf_iq2, ids, tmp_path / 'bat.npy',
                     ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1',
                      '--minimax-prefill-batch', '64'])
    d, per_row, top1, kl = _report(tok, bat, 'gpu grouped experts (iq2_xs)')
    assert np.median(per_row) < 2e-2, np.median(per_row)
    assert kl < 1e-4, kl
    assert top1 >= 0.97, top1


def test_expert_prefetch_does_not_change_logits(gguf, tmp_path):
    """The lookahead copies the same expert bytes earlier; the compute path is unchanged, so this is exact."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(32)
    base, _, _ = _run(gguf, ids, tmp_path / 'base.npy',
                      ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-tiers',
                       '--minimax-vram-slots-per-layer', '2', '--no-minimax-adaptive'])
    got, _, _ = _run(gguf, ids, tmp_path / 'pf.npy',
                     ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-tiers',
                      '--minimax-vram-slots-per-layer', '2', '--no-minimax-adaptive',
                      '--minimax-expert-prefetch', '2'])
    _report(base, got, 'prefetch vs none')
    # Same slots hold the same experts, so the routes and the kernels are identical; the extra lookahead router pass
    # leaves last-bit differences in the shared workspace (measured max 9.5e-7).
    assert np.abs(base - got).max() < 3e-6
    assert (base.argmax(axis=1) == got.argmax(axis=1)).all()


def test_serve_prefill_batch_matches_token_prefill(gguf, tmp_path):
    """End to end: the same greedy continuation after a 40-token prompt, before and after batching."""
    ids = _ids(40)
    prompt = ' '.join(str(i) for i in ids)

    def gen(extra):
        cmd = [EXE, '--model', str(gguf), '--serve', '--minimax-allow-nonstandard',
               '--max-context', '512', '--cpu-threads', '4', *extra]
        text = f'GEN 8 temperature=0 {prompt}\nSTOP\n'
        p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=900)
        toks = [line.split()[1] for line in p.stdout.splitlines() if line.startswith('T ')]
        assert p.returncode == 0, p.stderr
        assert len(toks) == 8, p.stdout + p.stderr
        return toks

    a = gen(['--minimax-dense-layers-gpu', '0'])
    b = gen(['--minimax-dense-layers-gpu', '0', '--minimax-prefill-batch', '32'])
    assert a == b
