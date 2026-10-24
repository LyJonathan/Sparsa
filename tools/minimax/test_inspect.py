"""P2a loader tests: the synthetic `minimax-m3` GGUF, the inspect table, and the serve skeleton.

Needs a built `strata-minimax.exe` (tools/minimax/build_windows.ps1 writes it to build/); the tests skip without it.
Run with: python -m pytest tools/minimax/test_inspect.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(0, str(HERE))
from make_synthetic_gguf import GEOMETRY, build  # noqa: E402


def _exe():
    candidates = [os.environ.get('STRATA_MINIMAX_EXE'),
                  str(ROOT / 'build' / 'strata-minimax.exe'),
                  str(ROOT / 'build' / 'strata-minimax'),
                  str(ROOT.parent / 'Strata-MiniMax' / 'build' / 'strata-minimax.exe')]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


EXE = _exe()
pytestmark = pytest.mark.skipif(EXE is None, reason='strata-minimax is not built (run tools/minimax/build_windows.ps1)')


def run(args, stdin=None, timeout=120):
    return subprocess.run([EXE, *args], input=stdin, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=timeout)


def test_inspect_synthetic(tmp_path):
    gguf = tmp_path / 'synthetic.gguf'
    build(gguf)
    p = run(['--model', str(gguf), '--minimax-inspect'])
    assert p.returncode == 0, p.stderr
    assert 'MiniMax GGUF: 2 layers, embd 256' in p.stdout
    assert 'geometry: NOT MiniMax-M3.1-Flash' in p.stdout
    assert 'token_embd.weight' in p.stdout
    assert 'blk.0.ffn_gate.weight' in p.stdout          # layer 0 is the dense prefix
    assert 'blk.1.ffn_gate_exps.weight' in p.stdout      # layer 1 is MoE
    assert 'blk.0.indexer.q_proj.weight' in p.stdout     # M3.1: indexer on the dense layers too
    assert 'bytes by class:' in p.stdout
    for cls in ('dense', 'experts', 'embd', 'output', 'total'):
        assert cls in p.stdout
    # The class totals are the sum of that class's tensors, and the rows are all present.
    blk = [l for l in p.stdout.splitlines() if ' blk.' in l]
    assert len(blk) == 2 * 8 + 3 + 8 + 4 + 4      # base per layer + dense MLP + routed/shared + indexer x2
    assert sum(1 for l in p.stdout.splitlines()
               if ' token_embd.weight' in l or ' output.weight' in l or ' output_norm.weight' in l) == 3


def test_inspect_rejects_a_missing_tensor(tmp_path):
    gguf = tmp_path / 'missing.gguf'
    build(gguf, drop='blk.1.ffn_gate_exps.weight')
    p = run(['--model', str(gguf), '--minimax-inspect'])
    assert p.returncode == 1
    assert 'missing tensor blk.1.ffn_gate_exps.weight' in p.stderr


def test_inspect_rejects_a_wrong_shape(tmp_path):
    gguf = tmp_path / 'shape.gguf'
    build(gguf, bad_shape=True)
    p = run(['--model', str(gguf), '--minimax-inspect'])
    assert p.returncode == 1
    assert 'blk.1.ffn_gate_exps.weight has shape' in p.stderr
    assert 'expected' in p.stderr


def test_serve_refuses_a_non_m3_1_file_without_the_development_flag(tmp_path):
    gguf = tmp_path / 'synthetic.gguf'
    build(gguf)
    p = run(['--model', str(gguf), '--serve'], stdin='QUIT\n')
    assert p.returncode == 1
    assert "this is not MiniMax-M3.1-Flash's geometry" in p.stderr
    assert 'READY' not in p.stdout


def test_serve_handshake_and_refusal(tmp_path):
    gguf = tmp_path / 'synthetic.gguf'
    build(gguf)
    p = run(['--model', str(gguf), '--serve', '--minimax-allow-nonstandard', '--max-context', '1024',
             '--not-an-engine-option', '700', '--batch', '1', '--cpu-threads', '1'],
            stdin='GEN 8 temperature=0.7 1,2,3\nGENI 4 embeddings.bin 1,2\nSTOP\nQUIT\n')
    assert p.returncode == 0, p.stderr
    assert 'INFO engine=minimax-m3' in p.stdout
    assert 'READY 1024 stop' in p.stdout
    # The zero-weight file really runs. Eight new tokens, then GENI is refused with no second DONE.
    tokens = [ln for ln in p.stdout.splitlines() if ln.startswith('T ')]
    dones = [ln for ln in p.stdout.splitlines() if ln.startswith('DONE ')]
    assert len(tokens) == 8, p.stdout
    assert len(dones) == 1, p.stdout
    assert dones[0].split()[5] == 'length'
    assert 'ERR minimax engine: GENI is not implemented (no vision encoder)' in p.stdout
    assert 'forward pass not implemented' not in p.stdout
    # Unknown serve flags are skipped, not fatal.
    assert 'ignoring --not-an-engine-option 700' in p.stderr


def test_vision_value_and_leftover_token_are_not_fatal(tmp_path):
    gguf = tmp_path / 'synthetic.gguf'
    build(gguf)
    p = run(['--model', str(gguf), '--minimax-inspect', '--vision', 'off'])
    assert p.returncode == 0, p.stderr
    served = run(['--model', str(gguf), '--serve', '--minimax-allow-nonstandard', '--vision', 'off',
                  '--max-context', '128', '--minimax-dense-layers-gpu', '0', 'not-a-flag'],
                 stdin='QUIT\n')
    assert served.returncode == 0, served.stderr
    assert 'READY 128 stop' in served.stdout
    assert "ignoring 'not-a-flag'" in served.stderr


def test_unasked_context_caps_at_32k_and_max_context_raises_it(tmp_path):
    gguf = tmp_path / 'wide.gguf'
    g = dict(GEOMETRY)
    g['ctx'] = 100000
    build(gguf, geometry=g)
    base = ['--model', str(gguf), '--serve', '--minimax-allow-nonstandard', '--minimax-dense-layers-gpu', '0']
    capped = run(base, stdin='QUIT\n')
    assert capped.returncode == 0, capped.stderr
    assert 'READY 32768 stop' in capped.stdout
    raised = run(base + ['--max-context', '40000'], stdin='QUIT\n')
    assert raised.returncode == 0, raised.stderr
    assert 'READY 40000 stop' in raised.stdout


def test_kv_alloc_failure_names_the_byte_count(tmp_path):
    gguf = tmp_path / 'huge.gguf'
    g = dict(GEOMETRY)
    # GEOMETRY: 2 layers, 2 kv heads, head 32, indexer 32. Host fp16 KV is 640 bytes per token.
    g['ctx'] = 300_000_000
    build(gguf, geometry=g)
    bytes_needed = 640 * g['ctx']
    p = run(['--model', str(gguf), '--serve', '--minimax-allow-nonstandard',
             '--minimax-dense-layers-gpu', '0', '--max-context', str(g['ctx'])],
            stdin='QUIT\n', timeout=30)
    assert p.returncode == 1, p.stdout
    assert str(bytes_needed) in p.stderr
    assert 'bytes are required' in p.stderr
    assert 'READY' not in p.stdout


def test_quit_joins_the_reader_while_stdin_stays_open(tmp_path):
    gguf = tmp_path / 'synthetic.gguf'
    build(gguf)
    proc = subprocess.Popen(
        [EXE, '--model', str(gguf), '--serve', '--minimax-allow-nonstandard',
         '--max-context', '128', '--minimax-dense-layers-gpu', '0'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='replace')
    out = []

    def drain():
        out.append(proc.stdout.read())

    thread = threading.Thread(target=drain)
    thread.start()
    proc.stdin.write('QUIT\n')
    proc.stdin.flush()
    try:
        rc = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        thread.join(timeout=5)
        raise AssertionError('QUIT did not exit while stdin was still open')
    proc.stdin.close()
    thread.join(timeout=5)
    assert rc == 0
    assert 'READY 128 stop' in ''.join(out)
