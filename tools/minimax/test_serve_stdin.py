"""tools/minimax/test_serve_stdin.py - `strata-minimax --serve` reads its GEN lines from a pipe AND from a redirected
file, and the two produce the same tokens.

The engine's stdin reader uses PeekNamedPipe / WaitForSingleObject on the console handle, neither of which sees a
redirected disk file: `strata-minimax --serve < ids.txt` used to spin forever (no READY output, no tokens, no exit) while
the same lines through a pipe worked.  This is the regression test for the FILE_TYPE_DISK branch in
src/minimax/serve.cpp.

It runs on the synthetic runnable model (tools/minimax/make_synthetic_gguf.py), so it needs no weights and no GPU:

    python -m pytest tools/minimax/test_serve_stdin.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))


def exe() -> str | None:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    return None


EXE = exe()
pytestmark = pytest.mark.skipif(EXE is None, reason='strata-minimax is not built')


@pytest.fixture(scope='module')
def model(tmp_path_factory) -> Path:
    from make_synthetic_gguf import build

    path = tmp_path_factory.mktemp('stdin') / 'tiny.gguf'
    build(path)
    return path


def tokens(ids: list[int], max_new: int) -> str:
    return f'GEN {max_new} temperature=0 ' + ' '.join(str(i) for i in ids) + '\nSTOP\n'


def run(model: Path, text: str, *, from_file: bool, tmp_path: Path) -> subprocess.CompletedProcess:
    cmd = [EXE, '--model', str(model), '--serve', '--minimax-allow-nonstandard', '--max-context', '256',
           '--cpu-threads', '2', '--gpus', '0']
    env = dict(os.environ)
    if from_file:
        ids_path = tmp_path / 'gen.txt'
        ids_path.write_text(text, encoding='ascii')
        with open(ids_path, 'rb') as f:
            return subprocess.run(cmd, stdin=f, capture_output=True, timeout=180, env=env)
    return subprocess.run(cmd, input=text.encode('ascii'), capture_output=True, timeout=180, env=env)


def started_tokens(out: bytes) -> list[int]:
    return [int(l.split()[1]) for l in out.decode('utf-8', 'replace').splitlines() if l.startswith('T ')]


def test_a_redirected_file_feeds_the_serve_loop(model: Path, tmp_path: Path):
    ids = [3, 20, 37, 54, 71, 88, 105, 122]
    p = run(model, tokens(ids, 4), from_file=True, tmp_path=tmp_path)
    out = p.stdout.decode('utf-8', 'replace')
    assert 'READY' in out, out[-800:]
    assert 'DONE' in out, out[-800:]
    assert len(started_tokens(p.stdout)) == 4, out[-800:]


def test_a_pipe_and_a_file_agree(model: Path, tmp_path: Path):
    ids = [3, 20, 37, 54, 71, 88, 105, 122]
    text = tokens(ids, 6)
    piped = run(model, text, from_file=False, tmp_path=tmp_path)
    filed = run(model, text, from_file=True, tmp_path=tmp_path)
    assert 'DONE' in piped.stdout.decode('utf-8', 'replace'), piped.stdout[-500:]
    assert started_tokens(piped.stdout) == started_tokens(filed.stdout)


def test_two_gen_lines_in_one_file(model: Path, tmp_path: Path):
    """A file holding two requests must serve both, in order, before the EOF ends the process."""
    ids = [3, 20, 37, 54]
    text = tokens(ids, 2) + tokens(ids, 3)
    p = run(model, text, from_file=True, tmp_path=tmp_path)
    out = p.stdout.decode('utf-8', 'replace')
    assert out.count('DONE') == 2, out[-800:]
    assert len(started_tokens(p.stdout)) == 5, out[-800:]
