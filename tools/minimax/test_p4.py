"""P4 gate: layer split over two GPUs, and the KV cache in VRAM vs pinned RAM.

The synthetic runnable model is tiny, so these run in seconds. Every case is compared against the same file on
one device, with the same KV placement: the split moves work, it must not change the math. The KV cases compare
the device attention in VRAM against the pinned-RAM cache with the per-step selected-block gather.

    python -m pytest tools/minimax/test_p4.py -q

Needs two CUDA devices for the split cases; they skip on a single-GPU machine.
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
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    assert p.returncode == 0, p.stderr
    return np.load(out), p.stdout, p.stderr


def _close(a, b, label, tol):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    scale = np.abs(a).max()
    print(f'{label}: max_abs={d.max():.3e} rel={d.max() / scale:.3e}')
    assert d.max() < tol, f'{label}: max abs {d.max():.3e} >= {tol:.3e}'


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


@pytest.fixture(scope='module')
def gguf(tmp_path_factory):
    p = tmp_path_factory.mktemp('p4') / 'runnable.gguf'
    build(p, geometry=RUNNABLE, weights='glorot')
    return p


def test_kv_in_ram_matches_kv_in_vram(gguf, tmp_path):
    ids = _ids(200)   # 25 indexer blocks of 8; top-16, so blocks are dropped (>17 blocks of context)
    base, _, _ = _run(gguf, ids, tmp_path / 'vram.npy',
                      ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1'])
    ram, _, _ = _run(gguf, ids, tmp_path / 'ram.npy',
                     ['--gpus', '0', '--minimax-kv', 'ram', '--minimax-dense-layers-gpu', '-1'])
    _close(base, ram, 'kv-ram-vs-vram', 1e-5)


def test_device_attention_matches_the_host_path(gguf, tmp_path):
    ids = _ids(200)
    host, _, _ = _run(gguf, ids, tmp_path / 'host.npy', ['--minimax-dense-layers-gpu', '0'])
    dev, _, _ = _run(gguf, ids, tmp_path / 'dev.npy',
                     ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '0'])
    # The device kernel accumulates in fp32 where the host path uses double, and softmax differs in the last bit.
    _close(host, dev, 'device-attn-vs-host', 2e-3)


@pytest.mark.skipif(NDEV < 2, reason='needs two CUDA devices')
def test_two_gpu_split_matches_single_gpu(gguf, tmp_path):
    ids = _ids(200)
    one, _, _ = _run(gguf, ids, tmp_path / 'one.npy',
                     ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '-1'])
    two, out, _ = _run(gguf, ids, tmp_path / 'two.npy',
                       ['--gpus', '0,1', '--layer-split', '2', '--minimax-kv', 'vram',
                        '--minimax-dense-layers-gpu', '-1'])
    # Same arithmetic, same order, two owners: this is an identity, not a tolerance.
    _close(one, two, 'split-vs-single (vram)', 1e-6)


@pytest.mark.skipif(NDEV < 2, reason='needs two CUDA devices')
def test_two_gpu_split_with_kv_in_ram(gguf, tmp_path):
    ids = _ids(200)
    one, _, _ = _run(gguf, ids, tmp_path / 'one-ram.npy',
                     ['--gpus', '0', '--minimax-kv', 'ram', '--minimax-dense-layers-gpu', '-1'])
    two, out, _ = _run(gguf, ids, tmp_path / 'two-ram.npy',
                       ['--gpus', '0,1', '--layer-split', '1', '--minimax-kv', 'ram',
                        '--minimax-dense-layers-gpu', '-1'])
    _close(one, two, 'split-vs-single (ram)', 1e-6)


@pytest.mark.skipif(NDEV < 2, reason='needs two CUDA devices')
def test_auto_split_reports_a_boundary(gguf, tmp_path):
    ids = _ids(12)
    got, out, err = _run(gguf, ids, tmp_path / 'auto.npy',
                         ['--gpus', '0,1', '--layer-split', 'auto', '--minimax-kv', 'auto',
                          '--minimax-dense-layers-gpu', '-1'])
    assert np.isfinite(got).all()
    line = [l for l in err.splitlines() if 'gpus=' in l]
    assert line, err
    assert 'gpus=0,1' in line[0]
    assert 'split=' in line[0]
    assert 'kv_space=' in line[0]
