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


def _run(gguf, ids, out, dense_gpu, last_k=0, env=None, max_context='512'):
    tok = out.with_suffix('.txt')
    tok.write_text(' '.join(str(i) for i in ids), encoding='utf-8')
    cmd = [EXE, '--model', str(gguf), '--minimax-allow-nonstandard',
           '--minimax-dense-layers-gpu', str(dense_gpu),
           '--minimax-eval', str(tok), '--minimax-dump-logits', str(out),
           '--max-context', str(max_context), '--cpu-threads', '1']
    if last_k:
        cmd += ['--minimax-last-k', str(last_k)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
                       env=env)
    assert p.returncode == 0, p.stderr
    got = np.load(out)
    return got, p.stdout


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


def _serve(gguf, stdin, extra=None, env=None):
    cmd = [EXE, '--model', str(gguf), '--minimax-allow-nonstandard', '--serve', '--greedy',
           '--minimax-dense-layers-gpu', '0', '--max-context', '512', '--cpu-threads', '1',
           *(extra or [])]
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=300, env=env)


def _tokens(stdout):
    """T ids from each DONE-terminated GEN, in order."""
    groups = []
    cur = []
    for line in stdout.splitlines():
        if line.startswith('T '):
            cur.append(int(line.split()[1]))
        elif line.startswith('DONE '):
            groups.append((cur, line.split()))
            cur = []
    return groups


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
        _check(expect, got, label, max_abs=1e-4, top1=1.0)


def test_gpu_dense_placement_stays_close_on_the_short_prompt(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    ids = _ids(12)
    expect = forward(load(gguf), ids)
    got, stdout = _run(gguf, ids, tmp_path / 'gpu.npy', dense_gpu=4)
    assert 'LOGITS' in stdout
    _check(expect, got, 'gpu-dense-select', max_abs=5e-2, top1=0.95)


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
    # quantizes the activation, so max-abs is not the gate. Top-1 and KL are.
    expect = forward(load(gguf), ids)
    for place, gpu in (('cpu', 0), ('gpu', 4)):
        got, _ = _run(gguf, ids, tmp_path / f'q8-{place}.npy', dense_gpu=gpu)
        _check(expect, got, f'q8-experts-{place}', top1=0.95, kl=1e-3)


# Block size of each dense type. K-quants need a row of 256; the runnable MLP width is 64, so those
# geometries widen dense_ff (and the shared-expert down row, which is exp_ff) to the block.
_DENSE_BLOCK = {'f16': 1, 'bf16': 1, 'q8_0': 32, 'q6_k': 256, 'q4_k': 256}
# f16/bf16 are a cast of the same weights. CPU dots the storage type in fp64; GPU dequants to fp32 and
# reduces in fp32. Integer quants also quantize the activation on the CPU vec_dot path, so the gate is
# top-1 and KL, not a tight max-abs. Bounds sit above the measured gaps (see the test printout).
# Q6_K on the CPU is the exception: vec_dot's activation quant flips a near-tie about once every 15
# tokens (11/12 = 0.917, 45/48 = 0.938). The GPU dequant path stays at top-1 1.0. 0.90 still fails a
# second miss on the 12-token prompt. KL stays under 2e-3.
_DENSE_BOUND = {
    'f16': dict(max_abs=2e-3, top1=1.0, kl=1e-6),
    'bf16': dict(max_abs=2e-3, top1=1.0, kl=1e-6),
    'q8_0': dict(top1=0.95, kl=1e-2),
    'q6_k': dict(top1=0.90, kl=1e-2),
    'q4_k': dict(top1=0.95, kl=1e-2),
}


def _quant_geometry(block, experts=False):
    g = dict(RUNNABLE)
    if g['dense_ff'] % block:
        g['dense_ff'] = block
    if experts or (g['exp_ff'] * g['shared']) % block:
        if g['exp_ff'] % block:
            g['exp_ff'] = block
    return g


@pytest.mark.parametrize('dtype', ['f16', 'bf16', 'q8_0', 'q6_k', 'q4_k'])
def test_dense_dtypes_match_reference_on_cpu_and_gpu(tmp_path, dtype):
    g = _quant_geometry(_DENSE_BLOCK[dtype])
    gguf = tmp_path / f'{dtype}.gguf'
    build(gguf, geometry=g, weights='glorot', dense_type=dtype)
    ids = _ids(12)
    expect = forward(load(gguf), ids)
    for place, gpu in (('cpu', 0), ('gpu', 4)):
        got, _ = _run(gguf, ids, tmp_path / f'{dtype}-{place}.npy', dense_gpu=gpu)
        assert got.shape == expect.shape
        _check(expect, got, f'dense-{dtype}-{place}', **_DENSE_BOUND[dtype])


def test_iq2_xs_experts_match_reference_on_cpu_and_gpu(tmp_path):
    # down-expert rows are exp_ff long; IQ2_XS blocks are 256. Uniform imatrix of ones.
    g = _quant_geometry(256, experts=True)
    gguf = tmp_path / 'iq2.gguf'
    build(gguf, geometry=g, weights='glorot', quant_experts='iq2_xs', pack_exe=EXE)
    ids = _ids(12)
    expect = forward(load(gguf), ids)
    for place, gpu in (('cpu', 0), ('gpu', 4)):
        got, _ = _run(gguf, ids, tmp_path / f'iq2-{place}.npy', dense_gpu=gpu)
        # Very coarse codes. The CPU path also quantizes the activation. Top-1 and KL, not max-abs.
        _check(expect, got, f'iq2xs-experts-{place}', top1=0.95, kl=1e-2)


def test_cpu_path_runs_with_no_visible_gpu(tmp_path):
    # CUDA_VISIBLE_DEVICES="" still enumerates a device on this Windows CUDA. -1 hides every device,
    # which is the check that dense_layers_gpu=0 never calls into the runtime.
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    ids = _ids(12)
    expect = forward(load(gguf), ids)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    got, _ = _run(gguf, ids, tmp_path / 'cpu-nogpu.npy', dense_gpu=0, env=env)
    _check(expect, got, 'cpu-no-gpu', max_abs=1e-4, top1=1.0)


def test_kv_reuse_matches_a_fresh_process(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    first = _serve(gguf, 'GEN 4 temperature=0 1,2,3\nQUIT\n')
    assert first.returncode == 0, first.stderr
    groups = _tokens(first.stdout)
    assert len(groups) == 1, first.stdout
    gen, done = groups[0]
    assert len(gen) == 4, first.stdout
    assert done[-1] == '0', done  # nothing cached on the first request
    prompt = [1, 2, 3, *gen]
    text = ','.join(str(t) for t in prompt)
    second = _serve(gguf, f'GEN 4 temperature=0 1,2,3\nGEN 2 temperature=0 {text}\nQUIT\n')
    assert second.returncode == 0, second.stderr
    groups = _tokens(second.stdout)
    assert len(groups) == 2, second.stdout
    assert groups[0][0] == gen
    # 3 prompt tokens + 3 forwarded samples. The 4th sample is held but not in the cache.
    assert groups[1][1][-1] == '6', groups[1][1]
    fresh = _serve(gguf, f'GEN 2 temperature=0 {text}\nQUIT\n')
    assert fresh.returncode == 0, fresh.stderr
    fresh_groups = _tokens(fresh.stdout)
    assert len(fresh_groups) == 1 and fresh_groups[0][1][-1] == '0'
    assert groups[1][0] == fresh_groups[0][0], (groups[1][0], fresh_groups[0][0])


def test_exact_cache_hit_reforward_matches_fresh(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    prompt = '1,2,3,4,5'
    held = _serve(gguf, f'GEN 0 temperature=0 {prompt}\nGEN 2 temperature=0 {prompt}\nQUIT\n')
    assert held.returncode == 0, held.stderr
    groups = _tokens(held.stdout)
    assert len(groups) == 2, held.stdout
    assert groups[0][0] == []
    # GEN 0 forwards every prompt token, so the cache length is the prompt and the hit re-forwards the last.
    assert groups[1][1][-1] == '5', groups[1][1]
    fresh = _serve(gguf, f'GEN 2 temperature=0 {prompt}\nQUIT\n')
    assert fresh.returncode == 0, fresh.stderr
    assert _tokens(fresh.stdout)[0][0] == groups[1][0]


def test_top_k_one_matches_greedy(tmp_path):
    gguf = tmp_path / 'runnable.gguf'
    build(gguf, geometry=RUNNABLE, weights='glorot')
    greedy = _serve(gguf, 'GEN 4 temperature=0 1,2,3\nQUIT\n')
    top = _serve(gguf, 'GEN 4 temperature=0.8 top_k=1 seed=1 1,2,3\nQUIT\n')
    assert greedy.returncode == 0, greedy.stderr
    assert top.returncode == 0, top.stderr
    assert _tokens(greedy.stdout)[0][0] == _tokens(top.stdout)[0][0]
