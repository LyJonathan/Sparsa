"""P7: `--minimax-kv vram` with every dense layer on the CPU.

The decode path (one token at a time) handled this: the KV is the device's and the attention kernel reads the
query from a device buffer, so nothing needs a host KV cache.  The *batched* prompt path did not: it chose the
device attention only when the layer's dense weights were also on the GPU (`dev_attn = gpu && kv_mode == 1`), so
with `--minimax-dense-layers-gpu 0` it took the host-attention branch.  `--minimax-kv vram` allocates no host K/V,
so that branch read a null pointer and the run died with an access violation about 50 s into a batched prefill on
the real model.

P7 makes the choice follow the KV's home (`dev_attn = kv_mode == 1`) and runs the layer's attention output
projection on the CPU when the layer's dense weights are not on the device.  `attend_host()` also refuses a
missing host K/V by name now, so a future call site cannot turn this class of bug back into a crash.

    python -m pytest tools/minimax/test_p7_kv_prefill.py -q
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
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax']:
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

COMMON = ['--minimax-allow-nonstandard', '--cpu-threads', '4']


def _ids(n):
    return [(i * 17 + 3) % RUNNABLE['vocab'] for i in range(n)]


def _cmd(gguf, ids, out, extra):
    tok = out.with_suffix('.txt')
    tok.write_text(' '.join(str(i) for i in ids), encoding='utf-8')
    return [EXE, '--model', str(gguf), *COMMON, '--minimax-eval', str(tok), '--minimax-dump-logits', str(out),
            '--max-context', '512', *extra]


@pytest.fixture(scope='module')
def gguf(tmp_path_factory):
    p = tmp_path_factory.mktemp('p7kv') / 'runnable.gguf'
    build(p, geometry=RUNNABLE, weights='glorot')
    return p


def test_vram_kv_with_cpu_dense_runs_batched_prefill(gguf, tmp_path):
    """The combination that crashed: --minimax-kv vram with every dense weight on the CPU, batched prefill.

    P7 routes the prompt pass's attention to the device whenever the KV is on the device, and runs that layer's
    output projection on the CPU, so the run completes instead of reading a null host K/V.
    """
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(64)
    got = tmp_path / 'cpudense.npy'
    p = subprocess.run(_cmd(gguf, ids, got,
                            ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '0',
                             '--minimax-prefill-batch', '8']),
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    assert 'prefill=batch' in p.stderr
    assert 'kv_space=vram' in p.stderr
    # the token path is the reference: same weights, same KV tier, one row at a time
    ref = tmp_path / 'cpudense_tok.npy'
    p = subprocess.run(_cmd(gguf, ids, ref,
                            ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '0']),
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    a, b = np.load(got), np.load(ref)
    assert float(np.abs(a - b).max()) < 1e-2, float(np.abs(a - b).max())


def test_vram_kv_with_cpu_dense_still_decodes(gguf, tmp_path):
    """The one-token path never needed host K/V: vram and host KV must still agree bit for bit."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(48)
    base = tmp_path / 'host.npy'
    p = subprocess.run(_cmd(gguf, ids, base, ['--gpus', '0', '--minimax-kv', 'host',
                                              '--minimax-dense-layers-gpu', '0']),
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    dev = tmp_path / 'vram.npy'
    p = subprocess.run(_cmd(gguf, ids, dev, ['--gpus', '0', '--minimax-kv', 'vram',
                                             '--minimax-dense-layers-gpu', '0']),
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    assert 'kv_space=vram' in p.stderr
    a, b = np.load(base), np.load(dev)
    # the device attention kernel and the host loop are not bit-identical (fp16 KV staging order differs); the
    # project's own bar for that comparison is 6.7e-5 max abs on this fixture
    assert float(np.abs(a - b).max()) < 1e-3, float(np.abs(a - b).max())


def test_vram_kv_with_one_dense_layer_on_gpu_runs_batched_prefill(gguf, tmp_path):
    """The supported arrangement: the batched device attention runs, so no refusal and no crash."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(64)
    p = subprocess.run(_cmd(gguf, ids, tmp_path / 'ok.npy',
                            ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-dense-layers-gpu', '1',
                             '--minimax-prefill-batch', '8']),
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    assert 'prefill=batch' in p.stderr


# ---- P7: the SSD tier's page-cache path and the one-layer-early prefetch must not change a single byte.  On this
# fixture the whole expert set fits in the process's working set, so every SSD-tier route takes the zero-copy path.

def _tiered(gguf, ids, out, extra):
    return subprocess.run(_cmd(gguf, ids, out,
                               ['--gpus', '0', '--minimax-tiers', '--minimax-vram-slots-per-layer', '0',
                                '--minimax-ram-experts-per-layer', '1', *extra]),
                          capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)


def test_ssd_zerocopy_is_bit_identical(gguf, tmp_path):
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(48)
    a, b = tmp_path / 'zc_off.npy', tmp_path / 'zc_on.npy'
    p = _tiered(gguf, ids, a, ['--minimax-ssd-zerocopy', 'off'])
    assert p.returncode == 0, p.stderr[-3000:]
    p = _tiered(gguf, ids, b, [])
    assert p.returncode == 0, p.stderr[-3000:]
    assert np.array_equal(np.load(a), np.load(b)), float(np.abs(np.load(a) - np.load(b)).max())


def test_ssd_prefetch_is_bit_identical(gguf, tmp_path):
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(48)
    a, b = tmp_path / 'pf_off.npy', tmp_path / 'pf_on.npy'
    p = _tiered(gguf, ids, a, ['--minimax-ssd-prefetch', 'off'])
    assert p.returncode == 0, p.stderr[-3000:]
    p = _tiered(gguf, ids, b, ['--minimax-ssd-prefetch', 'on'])
    assert p.returncode == 0, p.stderr[-3000:]
    assert np.array_equal(np.load(a), np.load(b)), float(np.abs(np.load(a) - np.load(b)).max())


def test_auto_ram_tier_reports_its_plan(gguf, tmp_path):
    """The auto budget is sized from the box and printed; the per-layer split sums to the reserved bytes."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    p = _tiered(gguf, _ids(16), tmp_path / 'plan.npy', [])
    assert p.returncode == 0, p.stderr[-3000:]
    line = [l for l in p.stderr.splitlines() if 'minimax-ram-tier plan' in l]
    assert line, p.stderr[-3000:]
    plan = dict(tok.split('=', 1) for tok in line[-1].split() if '=' in tok)
    assert plan['source'] in ('auto', '--memory-limit-mib', '--minimax-ram-experts-per-layer')
    assert float(plan['headroom'].split('GiB')[0]) >= 3.0
    lo, hi = plan['experts/layer'].split('..')
    assert 0 <= int(lo) <= int(hi)

