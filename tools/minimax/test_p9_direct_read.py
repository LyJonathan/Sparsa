"""P9: the streamed-read path under a RAM budget.

`--memory-limit-mib` bounds the pinned expert arena, but the default read path (`--minimax-prefill-read mmap`) grows
the OS page cache alongside it: the round-1 32 K run held 112.8 GB of working set with 0.3 GB of RAM left while the
tier was inside its 74 GiB budget.  With a budget the engine now reads a streamed expert unbuffered into the pinned
buffer instead of faulting it through the mapping.

The bytes are the same, so the logits must be **bit-identical** to the mmap path.  This test proves that on the
synthetic runnable model, on the CPU only (no CUDA, no expert tier), which is also the cheapest way to run it:

    python -m pytest tools/minimax/test_p9_direct_read.py -q
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
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build-lc2' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax.exe', ROOT / 'build-lc2' / 'strata-minimax',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    return None


EXE = _exe()
pytestmark = pytest.mark.skipif(EXE is None, reason='strata-minimax is not built')

COMMON = ['--minimax-allow-nonstandard', '--cpu-threads', '4', '--minimax-dense-layers-gpu', '0',
          '--minimax-prefill-batch', '64']


def _ids(n):
    return [(i * 17 + 3) % RUNNABLE['vocab'] for i in range(n)]


def _run(gguf, ids, out, extra):
    tok = out.with_suffix('.txt')
    tok.write_text(' '.join(str(i) for i in ids), encoding='utf-8')
    cmd = [EXE, '--model', str(gguf), *COMMON, '--minimax-eval', str(tok), '--minimax-dump-logits', str(out),
           '--max-context', '512', *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    if p.returncode != 0:
        raise AssertionError(f'engine failed: {p.stderr[-3000:]}')
    raw = np.load(out)
    return raw.reshape(-1, RUNNABLE['vocab'])


@pytest.fixture(scope='module')
def gguf(tmp_path_factory):
    p = tmp_path_factory.mktemp('p9read') / 'runnable.gguf'
    build(p, geometry=RUNNABLE, weights='glorot')
    return p


def test_direct_read_is_bit_identical(gguf, tmp_path):
    ids = _ids(64)
    mmap_logits = _run(gguf, ids, tmp_path / 'mmap.npy', [])
    # A budget turns the streamed reads unbuffered (the auto rule), and naming `direct` asks for it outright.
    budget_logits = _run(gguf, ids, tmp_path / 'budget.npy', ['--memory-limit-mib', '1024'])
    direct_logits = _run(gguf, ids, tmp_path / 'direct.npy',
                         ['--minimax-prefill-read', 'direct', '--memory-limit-mib', '1024'])
    assert np.array_equal(mmap_logits, budget_logits), \
        f'max abs {np.max(np.abs(mmap_logits - budget_logits))}'
    assert np.array_equal(mmap_logits, direct_logits), \
        f'max abs {np.max(np.abs(mmap_logits - direct_logits))}'


def test_direct_read_plan_line(gguf, tmp_path):
    out = tmp_path / 'plan.npy'
    tok = tmp_path / 'plan.txt'
    tok.write_text(' '.join(str(i) for i in _ids(64)), encoding='utf-8')
    cmd = [EXE, '--model', str(gguf), *COMMON, '--minimax-eval', str(tok), '--minimax-dump-logits', str(out),
           '--max-context', '512', '--memory-limit-mib', '1024']
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    assert 'minimax-prefill-read=direct' in p.stderr
    assert 'reads=direct' in p.stderr
