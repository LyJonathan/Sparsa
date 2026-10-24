"""The formats ported from ik_llama.cpp, end to end on the synthetic model.

Per type (IQ2_KT, IQ3_K, IQ4_KS, IQ3_KT, IQ2_KL):

* the engine's all-CPU run against `tiny_ref` on the *dequantized* weights (the stored bytes dequantized by the
  engine's own kernel, which is what the reference forward must read);
* the same model with every expert in the VRAM tier, i.e. through `minimax_moe_gu` / `minimax_moe_down` in
  `src/kernels/cuda/iq_kernels.cu`, against the CPU run (GPU dots use q8_1 activations, the CPU path Q8_K, so
  this is a top-1/KL bound like the IQ2_XS tier test).

The expert width is 512 (a whole number of 256-value super-blocks for every type) and hidden is 512, so neither
dimension is a geometry the older Qwen kernels were compiled for.

    python -m pytest tools/minimax/test_ikq.py -q
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

IKQ_TYPES = ('iq2_kt', 'iq3_k', 'iq4_ks', 'iq3_kt', 'iq2_kl')


def _exe():
    for c in (os.environ.get('STRATA_MINIMAX_EXE'), str(ROOT / 'build' / 'strata-minimax.exe'),
              str(ROOT / 'build-minimax' / 'strata-minimax.exe'), str(ROOT / 'build' / 'strata-minimax')):
        if c and Path(c).exists():
            return str(c)
    return None


EXE = _exe()
pytestmark = pytest.mark.skipif(EXE is None, reason='strata-minimax is not built (run tools/minimax/build_windows.ps1)')


def _ids(n):
    return [(i * 17 + 3) % RUNNABLE['vocab'] for i in range(n)]


def _run(gguf, ids, out, extra=()):
    tok = out.with_suffix('.txt')
    tok.write_text(' '.join(str(i) for i in ids), encoding='utf-8')
    cmd = [EXE, '--model', str(gguf), '--minimax-allow-nonstandard', '--minimax-dense-layers-gpu', '0',
           '--minimax-eval', str(tok), '--minimax-dump-logits', str(out), '--max-context', '512',
           '--cpu-threads', '1', *(extra or ())]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    assert p.returncode == 0, p.stderr
    return np.load(out), p.stdout, p.stderr


def _tiers(vram=8, ram=0, mem=64):
    return ['--minimax-tiers', '--minimax-vram-slots-per-layer', str(vram),
            '--minimax-ram-experts-per-layer', str(ram), '--memory-limit-mib', str(mem), '--minimax-stats']


def _vram_hits(stderr):
    line = None
    for ln in stderr.splitlines():
        if 'minimax-stats total' in ln:
            line = ln.split('minimax-stats total', 1)[1].strip()
    assert line is not None, stderr
    return int(dict(item.split('=', 1) for item in line.split())['vram_hits'])


@pytest.mark.parametrize('kind', IKQ_TYPES)
def test_ikq_experts_cpu_vs_ref_and_gpu(tmp_path, kind):
    g = dict(RUNNABLE)
    g['hidden'] = 512
    g['dense_ff'] = 512
    g['exp_ff'] = 512
    gguf = tmp_path / f'{kind}.gguf'
    ref_gguf = tmp_path / f'{kind}-ref.gguf'
    build(gguf, geometry=g, weights='glorot', quant_experts=kind, pack_exe=EXE, deq_copy=ref_gguf)
    assert ref_gguf.exists(), 'the f32 reference model was not written'

    ids = _ids(12)
    cpu, _, _ = _run(gguf, ids, tmp_path / 'cpu.npy')

    # 1. tiny_ref on the dequantized weights.
    ref = forward(load(ref_gguf), ids)
    st = compare(ref, cpu)
    print(f'{kind} cpu-vs-ref: max_abs={st["max_abs"]:.3e} kl={st["kl"]:.3e} top1={st["top1"]:.3f}')
    assert np.isfinite(cpu).all()
    # Quantized experts: the CPU vec_dot quantizes the activation too, so this is a tolerance, not equality.
    assert st['top1'] >= 0.95, st
    assert st['kl'] < 1e-2, st

    # 2. the CUDA path (all experts in VRAM).  GPU dots use q8_1 activations, the CPU path Q8_K: the bound is
    # the one the existing IQ2_XS tier test uses (measured 2.1e-2 max-abs, KL 1.3e-5 there).  KL decides; top-1
    # on a random model moves when two logits sit inside the activation-quantization noise (one token of twelve
    # for IQ2_KL), which is not a kernel error.
    tier, _, err = _run(gguf, ids, tmp_path / 'tier.npy', _tiers())
    st = compare(cpu, tier)
    print(f'{kind} gpu-vs-cpu: max_abs={st["max_abs"]:.3e} kl={st["kl"]:.3e} top1={st["top1"]:.3f}')
    assert np.isfinite(tier).all()
    assert st['top1'] >= 0.9, st
    assert st['kl'] < 1e-2, st
    assert _vram_hits(err) > 0, err
