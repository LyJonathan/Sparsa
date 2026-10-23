"""P3 tier tests: VRAM / pinned RAM / SSD routed-expert tiers against the all-CPU path.

The small runnable model has one MoE layer with 8 experts (top-4).  A budget of 2 VRAM slots and 3 pinned
RAM slots leaves three experts on the SSD, so one run exercises all three tiers.  The profile test saves
the routing the engine observed, reads the STRP file back, and runs from it.

    python -m pytest tools/minimax/test_tiers.py -q

The tier run needs a CUDA device (the VRAM kernel); the CPU baseline does not.
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
from tiny_ref import compare  # noqa: E402

import make_profile  # noqa: E402


def _exe():
    for c in (os.environ.get('STRATA_MINIMAX_EXE'), str(ROOT / 'build' / 'strata-minimax.exe'),
              str(ROOT / 'build' / 'strata-minimax')):
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
           '--minimax-eval', str(tok), '--minimax-dump-logits', str(out),
           '--max-context', '512', '--cpu-threads', '1', *(extra or ())]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    assert p.returncode == 0, p.stderr
    return np.load(out), p.stdout, p.stderr


def _stats(stderr):
    """The final `INFO minimax-stats total ...` line as a dict of integers."""
    line = None
    for ln in stderr.splitlines():
        if 'minimax-stats total' in ln:
            line = ln.split('minimax-stats total', 1)[1].strip()
    assert line is not None, stderr
    return {k: int(v) for k, v in (item.split('=', 1) for item in line.split()) if k != 'tiers'}


def _tiers(vram=2, ram=3, mem=256, stats=True):
    args = ['--minimax-tiers', '--minimax-vram-slots-per-layer', str(vram),
            '--minimax-ram-experts-per-layer', str(ram), '--memory-limit-mib', str(mem)]
    if stats:
        args.append('--minimax-stats')
    return args


def _check(reference, got, label, max_abs=None, top1=None, kl=None):
    stats = compare(reference, got)
    print(f'{label}: max_abs={stats["max_abs"]:.3e} kl={stats["kl"]:.3e} top1={stats["top1"]:.3f}')
    assert np.isfinite(got).all()
    if max_abs is not None:
        assert stats['max_abs'] < max_abs, stats
    if top1 is not None:
        assert stats['top1'] >= top1, stats
    if kl is not None:
        assert stats['kl'] < kl, stats
    return stats


def test_all_three_tiers_match_the_cpu_path(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    ids = _ids(40)
    base, _, _ = _run(gguf, ids, tmp_path / 'cpu.npy')
    tier, _, err = _run(gguf, ids, tmp_path / 'tier.npy', _tiers())
    _check(base, tier, 'tier-vram2-ram3-ssd', max_abs=2e-3, top1=1.0, kl=1e-6)
    st = _stats(err)
    print('tier stats:', st)
    assert st['vram_hits'] > 0 and st['ram_hits'] > 0, st
    assert st['ssd_reads'] > 0 and st['ssd_bytes'] > 0, st
    assert st['ssd_reads'] == st['ssd_bytes'] // (3 * 64 * 256 * 4), st
    assert st['swaps'] > 0, st   # 40 tokens cross two decayed-count rebalances


def test_profile_save_load_round_trip(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    ids = _ids(10)
    base, _, _ = _run(gguf, ids, tmp_path / 'cpu.npy')
    prof, counts = tmp_path / 'profile.bin', tmp_path / 'counts.txt'
    _run(gguf, ids, tmp_path / 'saved.npy', ['--minimax-profile-save', str(prof),
                                             '--minimax-routing-counts', str(counts)])
    n_moe, n_expert, ranked = make_profile.read_profile(prof)
    assert (n_moe, n_expert) == (1, 8)
    assert len(ranked) == 8 and ranked[0][0] == 0
    # The Python builder writes the same bytes from the text table.
    again = tmp_path / 'again.bin'
    make_profile.write_profile(again, ranked, n_moe, n_expert)
    assert prof.read_bytes() == again.read_bytes()
    # And the engine can run from it: the fill changes which tier an expert lives in, not the math.
    tier, _, err = _run(gguf, ids, tmp_path / 'tier.npy',
                        ['--minimax-profile', str(prof), '--minimax-vram-slots-per-layer', '2',
                         '--minimax-ram-experts-per-layer', '3', '--memory-limit-mib', '256', '--minimax-stats'])
    _check(base, tier, 'profile-load', max_abs=2e-3, top1=1.0, kl=1e-6)
    st = _stats(err)
    assert st['vram_hits'] > 0 and st['ram_hits'] > 0, st


def test_iq2_xs_experts_on_the_gpu_tier(tmp_path):
    g = dict(RUNNABLE)
    g['exp_ff'] = 256   # IQ2_XS blocks are 256 wide
    gguf = tmp_path / 'iq2.gguf'
    build(gguf, geometry=g, weights='glorot', quant_experts='iq2_xs', pack_exe=EXE)
    ids = _ids(12)
    base, _, _ = _run(gguf, ids, tmp_path / 'cpu.npy')
    tier, _, err = _run(gguf, ids, tmp_path / 'tier.npy', _tiers())
    # GPU iq kernels use q8_1 activations, the CPU path ggml's Q8_K: compare top-1 and KL, like test_forward.
    _check(base, tier, 'iq2xs-gpu-tier', top1=0.95, kl=1e-2)
    st = _stats(err)
    assert st['vram_hits'] > 0, st


def test_q8_0_experts_on_the_gpu_tier(tmp_path):
    gguf = tmp_path / 'q8.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot', quant_experts='q8_0', pack_exe=EXE)
    ids = _ids(12)
    base, _, _ = _run(gguf, ids, tmp_path / 'cpu.npy')
    tier, _, err = _run(gguf, ids, tmp_path / 'tier.npy', _tiers())
    _check(base, tier, 'q8-gpu-tier', max_abs=2e-3, top1=1.0, kl=1e-6)
    st = _stats(err)
    assert st['vram_hits'] > 0, st
