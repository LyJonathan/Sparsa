"""P8: long context.  KV quantization, the multithreaded host attention, and the staged prompt pass.

Three opt-in paths, each measured against the path it replaces on the same fixture:

* `--minimax-host-attn fast` — the P2 host attention allocated 65 vectors per token per layer and ran the
  indexer score loop (4 heads x every key) on one thread.  The rewrite keeps the same arithmetic order and
  splits the independent work over the pool, so its logits must be **bit-identical** to `legacy`.
* `--minimax-kv-quant q8_0` — halves the host/pinned K/V.  It cannot be bit-identical (it is a quantization),
  so the test records its measured drift and asserts it stays inside the project's quantized-path bar.
* `--minimax-prefill-kv auto` (default) — with the K/V in pinned RAM the batched prompt pass used to run
  `attend_host` per token, which is ~1 s/token at 32 K.  It now stages one layer's K/V on the card and runs
  the device MSA kernel; the result must match the token path within the device/host bar.

    python -m pytest tools/minimax/test_p8_longctx.py -q
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
    p = tmp_path_factory.mktemp('p8lc') / 'runnable.gguf'
    build(p, geometry=RUNNABLE, weights='glorot')
    return p


def _run(gguf, ids, out, extra):
    p = subprocess.run(_cmd(gguf, ids, out, extra), capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stderr


def _plan(stderr):
    line = [l for l in stderr.splitlines() if 'minimax-kv plan' in l]
    assert line, stderr[-3000:]
    return dict(tok.split('=', 1) for tok in line[-1].split() if '=' in tok)


# ---- the multithreaded host attention ---------------------------------------------------------------------

def test_host_attn_fast_is_bit_identical(gguf, tmp_path):
    """The rewrite is a scheduling change only: same dots, same order, same softmax -> identical logits."""
    ids = _ids(96)
    a, b = tmp_path / 'fast.npy', tmp_path / 'legacy.npy'
    cpu = ['--gpus', '0', '--minimax-dense-layers-gpu', '0']
    sa = _run(gguf, ids, a, [*cpu, '--minimax-host-attn', 'fast'])
    _run(gguf, ids, b, [*cpu, '--minimax-host-attn', 'legacy'])
    assert 'host_attn=fast' in sa
    x, y = np.load(a), np.load(b)
    assert np.array_equal(x, y), float(np.abs(x - y).max())


def test_host_attn_default_is_fast_and_matches_legacy(gguf, tmp_path):
    """The unasked default is the fast path (it is byte-identical, so nothing else can depend on it)."""
    ids = _ids(48)
    a, b = tmp_path / 'def.npy', tmp_path / 'leg.npy'
    cpu = ['--gpus', '0', '--minimax-dense-layers-gpu', '0']
    s = _run(gguf, ids, a, cpu)
    assert 'host_attn=fast' in s
    _run(gguf, ids, b, [*cpu, '--minimax-host-attn', 'legacy'])
    assert np.array_equal(np.load(a), np.load(b))


def _kv_kl(a, b):
    """Mean KL(p_f16 || p_quant) over the rows, in nats, plus the top-1 agreement."""
    def logsm(x):
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    la, lb = logsm(a.astype(np.float64)), logsm(b.astype(np.float64))
    kl = (np.exp(la) * (la - lb)).sum(axis=1)
    return float(kl.mean()), float((a.argmax(1) == b.argmax(1)).mean())


# ---- q8_0 K/V ----------------------------------------------------------------------------------------------

def test_kv_q8_0_halves_the_cache_and_stays_close(gguf, tmp_path):
    """q8_0 blocks are 34 B / 32 values against 64 B: the plan line must say so, and the logits must hold."""
    ids = _ids(96)
    a, b = tmp_path / 'f16.npy', tmp_path / 'q8.npy'
    cpu = ['--gpus', '0', '--minimax-dense-layers-gpu', '0']
    pa = _plan(_run(gguf, ids, a, cpu))
    pb = _plan(_run(gguf, ids, b, [*cpu, '--minimax-kv-quant', 'q8_0']))
    assert pa['quant'] == 'f16'
    assert pb['quant'] == 'q8_0'
    fa, fb = float(pa['kv'].split('GiB')[0]), float(pb['kv'].split('GiB')[0])
    # K and V shrink to 17/32 of their size and the indexer keys stay f16; on the real geometry that is 0.58 of the
    # f16 cache, and this fixture's indexer is a larger share (idx_dim 32 vs head_dim 32).
    assert 0.4 < fb / fa < 0.7, (fa, fb)
    kl, top1 = _kv_kl(np.load(a), np.load(b))
    # measured on this fixture (4 layers, head_dim 32 -> one 32-value block per head, the worst case for q8_0):
    # mean KL 6.0e-4 nats over 96 rows (1.8e-4 over 64), top-1 94-97%.  The bar is the project's quantized-path
    # KL: the f16 prefill activation was accepted at 1.7e-3.
    assert kl < 1.7e-3, kl
    assert top1 >= 0.9, top1


def test_kv_q8_0_on_the_ram_tier(gguf, tmp_path):
    """The pinned-RAM tier keeps indexer keys on the card and the main K/V quantized in host memory."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(64)
    a, b = tmp_path / 'ram16.npy', tmp_path / 'ram8.npy'
    ram = ['--gpus', '0', '--minimax-dense-layers-gpu', '0', '--minimax-kv', 'ram']
    _run(gguf, ids, a, ram)
    s = _run(gguf, ids, b, [*ram, '--minimax-kv-quant', 'q8_0'])
    assert 'kv_space=ram' in s
    assert _plan(s)['quant'] == 'q8_0'
    kl, top1 = _kv_kl(np.load(a), np.load(b))
    assert kl < 5e-4, kl
    assert top1 >= 0.9, top1


def test_kv_q8_0_is_refused_for_the_vram_tier(gguf, tmp_path):
    """The device MSA kernels read fp16 rows; the request must be named, not silently ignored."""
    ids = _ids(16)
    p = subprocess.run(_cmd(gguf, ids, tmp_path / 'no.npy',
                            ['--gpus', '0', '--minimax-kv', 'vram', '--minimax-kv-quant', 'q8_0']),
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode != 0
    assert 'q8_0 covers the host and pinned-RAM KV tiers' in p.stderr


# ---- the staged prompt pass -------------------------------------------------------------------------------

def test_staged_prefill_matches_the_token_path(gguf, tmp_path):
    """--minimax-kv ram + batched prefill now runs MSA on the card instead of attend_host per token."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(96)
    a, b = tmp_path / 'stage.npy', tmp_path / 'tok.npy'
    ram = ['--gpus', '0', '--minimax-dense-layers-gpu', '0', '--minimax-kv', 'ram']
    s = _run(gguf, ids, a, [*ram, '--minimax-prefill-batch', '8'])
    assert 'prefill=batch' in s
    assert 'prefill_kv=stage' in s
    _run(gguf, ids, b, ram)
    x, y = np.load(a), np.load(b)
    assert float(np.abs(x - y).max()) < 1e-2, float(np.abs(x - y).max())


def test_prefill_kv_off_keeps_the_host_path(gguf, tmp_path):
    """Without a device KV tier the host prompt pass is the supported one; `off` must keep it, not crash."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(48)
    s = _run(gguf, ids, tmp_path / 'off.npy',
             ['--gpus', '0', '--minimax-dense-layers-gpu', '0', '--minimax-kv', 'host',
              '--minimax-prefill-batch', '8', '--minimax-prefill-kv', 'off'])
    assert 'prefill_kv=host' in s


def test_ram_tier_without_staging_names_the_refusal(gguf, tmp_path):
    """In the pinned-RAM tier the indexer keys are device-only, so the host prompt pass cannot run: say so."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(48)
    p = subprocess.run(_cmd(gguf, ids, tmp_path / 'refused.npy',
                            ['--gpus', '0', '--minimax-dense-layers-gpu', '0', '--minimax-kv', 'ram',
                             '--minimax-prefill-batch', '8', '--minimax-prefill-kv', 'off']),
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode != 0
    assert 'needs the indexer keys in host memory' in p.stderr


# ---- the plan -----------------------------------------------------------------------------------------------

def test_auto_plan_names_the_placement_and_the_quant(gguf, tmp_path):
    ids = _ids(16)
    s = _run(gguf, ids, tmp_path / 'plan.npy', ['--gpus', '0', '--minimax-dense-layers-gpu', '0'])
    plan = _plan(s)
    assert plan['placement'] in ('host', 'vram', 'ram')
    assert plan['quant'] in ('f16', 'q8_0')
    assert plan['indexer'] == 'f16'
    assert int(plan['rows'].split('B/layer')[0]) > 0


def test_auto_quant_stays_f16_when_it_fits(gguf, tmp_path):
    """`auto` is a capacity decision, not a default: a fixture that fits keeps the byte-identical fp16 cache."""
    if NDEV < 1:
        pytest.skip('needs a CUDA device')
    ids = _ids(16)
    s = _run(gguf, ids, tmp_path / 'auto.npy',
             ['--gpus', '0', '--minimax-dense-layers-gpu', '0', '--minimax-kv', 'ram',
              '--minimax-kv-quant', 'auto'])
    assert _plan(s)['quant'] == 'f16'
