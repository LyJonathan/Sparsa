"""tools/minimax/test_dspark.py - the DSpark draft head's pack and the speculative loop's correctness gates.

The pack cases need only the checkpoint and the pack (no GPU): they check the geometry the engine reads, and they
decode one MXFP8 tensor twice - once from the checkpoint's own F8_E4M3 + E8M0 scales, once from the Q8_0 blocks in
the pack - so a wrong block size, a transposed row or a missed scale shows up as a number that is not close.

The engine cases (a GPU, the slice or the full model) are the ones that matter for the deliverable:

  * `test_spec_matches_plain` - the same greedy prompt with and without --speculative, token for token.
  * `test_corrupt_drafts_still_match` - STRATA_DSPARK_CORRUPT replaces every draft with a wrong token, so every round
    is rejected and its KV rows are rolled back; the tokens must STILL equal plain decode.  This is the rollback
    gate (nothing else in the engine moves n_past backwards).
  * `test_acceptance_and_speed` - prints (does not assert) the acceptance rate and the two decode rates.

    python -m pytest tools/minimax/test_dspark.py -q            # pack cases only, no GPU
    STRATA_DSPARK_MODEL=<model.gguf> STRATA_DSPARK_PACK=<pack.gguf> STRATA_DSPARK_IDS=<ids.txt> \\
        python -m pytest tools/minimax/test_dspark.py -q
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

CKPT = pathlib_ckpt = None


def _ckpt_dir() -> Path | None:
    p = os.environ.get('STRATA_DSPARK_CKPT')
    if p:
        return Path(p)
    p = ROOT.parent / 'models' / 'MiniMax-M3.1-Flash' / 'dspark'
    return p if p.exists() else None


def pack_path() -> Path | None:
    p = os.environ.get('STRATA_DSPARK_PACK')
    if p:
        return Path(p)
    p = ROOT.parent / 'models' / 'MiniMax-M3.1-Flash' / 'dspark-q8_0.gguf'
    return p if p.exists() else None


def exe() -> str | None:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    return None


class Pack:
    """The smallest reader that answers the questions the engine's own reader answers."""

    def __init__(self, path: Path):
        buf = path.read_bytes()
        magic, ver, nt, nkv = struct.unpack_from('<IIQQ', buf, 0)
        assert magic == 0x46554747 and ver == 3
        pos = 24

        def rstr(p):
            n, = struct.unpack_from('<Q', buf, p)
            return buf[p + 8:p + 8 + n].decode(), p + 8 + n

        def rval(p, t):
            if t == 8:
                return rstr(p)
            if t == 9:   # array: element type then count
                et, n = struct.unpack_from('<IQ', buf, p)
                p += 12
                items = []
                for _ in range(n):
                    v, p = rval(p, et)
                    items.append(v)
                return items, p
            fmt = {4: 'I', 5: 'i', 6: 'f', 7: 'B', 10: 'Q'}[t]
            v, = struct.unpack_from('<' + fmt, buf, p)
            return v, p + struct.calcsize(fmt)

        self.meta = {}
        for _ in range(nkv):
            k, pos = rstr(pos)
            t, = struct.unpack_from('<I', buf, pos)
            pos += 4
            v, pos = rval(pos, t)
            self.meta[k] = v
        self.t = {}
        for _ in range(nt):
            name, pos = rstr(pos)
            nd, = struct.unpack_from('<I', buf, pos)
            pos += 4
            dims = struct.unpack_from('<%dQ' % nd, buf, pos)
            pos += 8 * nd
            typ, off = struct.unpack_from('<IQ', buf, pos)
            pos += 12
            self.t[name] = (dims, typ, off)
        self.align = 32
        self.data = (pos + 31) // 32 * 32
        self.buf = buf

    def rows(self, name) -> np.ndarray:
        """The tensor as F32, rows x ncols (GGUF dims reversed), decoded with ggml's own block rules."""
        dims, typ, off = self.t[name]
        n = int(np.prod(dims))
        ncols = int(dims[0])
        start = self.data + off
        if typ == 1:      # F16
            return np.frombuffer(self.buf[start:start + n * 2], np.float16).astype(np.float32).reshape(-1, ncols)
        if typ == 8:      # Q8_0: fp16 d + 32 int8
            raw = np.frombuffer(self.buf[start:start + (n // 32) * 34], np.uint8).reshape(-1, 34)
            d = raw[:, :2].copy().view(np.float16).astype(np.float32)
            q = raw[:, 2:].copy().view(np.int8).astype(np.float32)
            return (q * d).reshape(-1, ncols)
        if typ == 2:      # Q4_0: value[j] = (lo nibble - 8) d, value[j + 16] = (hi nibble - 8) d
            raw = np.frombuffer(self.buf[start:start + (n // 32) * 18], np.uint8).reshape(-1, 18)
            d = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
            qs = raw[:, 2:]
            q = np.empty((raw.shape[0], 32), np.float32)
            q[:, :16] = (qs & 15).astype(np.float32) - 8
            q[:, 16:] = (qs >> 4).astype(np.float32) - 8
            return (q * d[:, None]).reshape(-1, ncols)
        if typ == 6:      # Q5_0: d = max / -16, the 5th bits at qh[j] and qh[j + 16], values in halves
            raw = np.frombuffer(self.buf[start:start + (n // 32) * 22], np.uint8).reshape(-1, 22)
            d = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
            qh = raw[:, 2:6].copy().view(np.uint32).reshape(-1)
            qs = raw[:, 6:]
            q = np.empty((raw.shape[0], 32), np.float32)
            for j in range(16):
                xh0 = ((qh >> np.uint32(j)) << np.uint32(4)) & np.uint32(0x10)
                xh1 = (qh >> np.uint32(j + 12)) & np.uint32(0x10)   # ggml: bit j + 16
                q[:, j] = ((qs[:, j] & 15) | xh0).astype(np.float32) - 16
                q[:, 16 + j] = ((qs[:, j] >> 4) | xh1).astype(np.float32) - 16
            return (q * d[:, None]).reshape(-1, ncols)
        raise AssertionError('type %d' % typ)


def _ckpt_tensor(ck, name) -> np.ndarray:
    """One MXFP8 / BF16 tensor from the checkpoint, decoded to F32 (the reference the pack must agree with)."""
    raw, dt, sh = ck.raw(name)
    if dt == 'BF16':
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(sh)
    if dt == 'F8_E4M3':
        tbl = np.zeros(256, np.float32)
        for c in range(256):
            s, e, m = c >> 7, (c >> 3) & 15, c & 7
            v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
            tbl[c] = -v if s else v
        sraw, sdt, ssh = ck.raw(name[:-len('weight')] + 'weight_scale_inv')
        sc = np.frombuffer(sraw, np.uint8).reshape(ssh)
        per = sh[1] // ssh[1]
        table = np.repeat(np.exp2(sc.astype(np.float32) - 127.0), per, axis=1)[:, :sh[1]]
        return tbl[np.frombuffer(raw, np.uint8).reshape(sh)] * table
    raise AssertionError(dt)


@pytest.mark.skipif(pack_path() is None, reason='no DSpark pack (see tools/minimax/dspark_pack.py)')
def test_pack_geometry():
    pk = Pack(pack_path())
    m = pk.meta
    assert m['general.architecture'] == 'minimax-dspark'
    n_embd = m['dspark.embedding_length']
    n_layers = m['dspark.block_count']
    taps = m['dspark.tap_layers']
    assert taps == [3, 17, 31, 45, 59], taps
    assert m['dspark.block_size'] >= 1
    for name in ['fc.weight', 'hidden_norm.weight', 'final_norm.weight', 'markov_w1.weight', 'markov_w2.weight',
                 'confidence_head.weight', 'confidence_head.bias']:
        assert name in pk.t, name
    dims, typ, _ = pk.t['fc.weight']
    assert dims == (len(taps) * n_embd, n_embd), dims
    dims, _, _ = pk.t['hidden_norm.weight']
    assert dims == (n_embd,), dims
    for l in range(n_layers):
        for suffix in ['input_layernorm.weight', 'post_attention_layernorm.weight', 'self_attn.q_proj.weight',
                       'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
                       'self_attn.q_norm.weight', 'self_attn.k_norm.weight', 'mlp.gate_proj.weight',
                       'mlp.up_proj.weight', 'mlp.down_proj.weight']:
            name = f'layers.{l}.{suffix}'
            assert name in pk.t, name
    # every projection shares one type (the pack's --type: Q4_0/Q5_0 for a 4-5 bit drafter, Q8_0 for the reference);
    # the Markov pair and the confidence head stay F16 because the host reads their rows one at a time
    kinds = {typ for name, (dims, typ, _) in pk.t.items()
             if len(dims) > 1 and not name.startswith(('markov_', 'confidence_'))}
    assert len(kinds) == 1 and kinds <= {2, 6, 8}, kinds
    for name, (dims, typ, _) in pk.t.items():
        if len(dims) == 1 or name.startswith(('markov_', 'confidence_')):
            assert typ == 1, (name, typ)


@pytest.mark.skipif(pack_path() is None or _ckpt_dir() is None, reason='no pack or no checkpoint')
def test_pack_matches_the_checkpoint():
    """One expert-free layer's projections, decoded from the checkpoint and from the pack: Q8_0 error only."""
    sys.path.insert(0, str(HERE))
    from dspark_pack import Checkpoint  # noqa: E402

    ck = Checkpoint(_ckpt_dir())
    pk = Pack(pack_path())
    prefix = 'language_model.model.dspark.layers.0.decoder_layer.'
    for suffix, name in [('self_attn.q_proj.weight', 'layers.0.self_attn.q_proj.weight'),
                         ('mlp.gate_proj.weight', 'layers.0.mlp.gate_proj.weight'),
                         ('self_attn.o_proj.weight', 'layers.0.self_attn.o_proj.weight')]:
        ref = _ckpt_tensor(ck, prefix + suffix)
        got = pk.rows(name)
        assert got.shape == ref.shape, (name, got.shape, ref.shape)
        # Q8_0's own error is ~0.5% RMS on activations of this shape; anything bigger means a wrong block size, a
        # transposed row or a missed E8M0 scale (an unscaled decode is off by 2**12)
        rms = float(np.sqrt(((got.astype(np.float64) - ref) ** 2).mean()) / np.sqrt((ref.astype(np.float64) ** 2).mean()))
        # Q8_0 ~0.5% RMS, Q5_0 ~3.5%, Q4_0 ~9%: anything above the type's own noise means the block layout is wrong
        wide = max(int(d) for d in pk.t[name][0])
        limit = {8: 0.01, 6: 0.06, 2: 0.14}[pk.t[name][1]]
        assert rms < limit, (name, rms, limit)
        # a transposed or shifted read would also break row by row, so check the shape of the error too
        row = np.linalg.norm(got.astype(np.float64) - ref, axis=1) / np.linalg.norm(ref.astype(np.float64), axis=1)
        assert float(row.max()) < 2.5 * limit, (name, float(row.max()))


def _ids(n: int, vocab: int = 200000, seed: int = 11) -> list[int]:
    x = seed
    out = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x % vocab)
    return out


def _run(model: str, ids: list[int], max_new: int, extra: list[str], ctx: int = 4096, env=None):
    cmd = [exe(), '--model', model, '--serve', '--max-context', str(ctx), *extra]
    text = f'GEN {max_new} temperature=0 ' + ' '.join(map(str, ids)) + '\nSTOP\n'
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=3600, env=e)
    if p.returncode != 0:
        raise AssertionError(p.stderr[-3000:])
    toks = [int(l.split()[1]) for l in p.stdout.splitlines() if l.startswith('T ')]
    done = [l for l in p.stdout.splitlines() if l.startswith('DONE ')]
    assert done, p.stdout[-1500:]
    f = done[0].split()
    spec = [l for l in p.stderr.splitlines() if l.startswith('INFO minimax-spec ')]
    return toks, {'gen': int(f[1]), 'prompt': int(f[2]), 'prompt_ms': float(f[3]), 'decode_ms': float(f[4]),
                  'spec': spec[-1] if spec else ''}, p.stderr


def _model() -> str | None:
    return os.environ.get('STRATA_DSPARK_MODEL')


ENGINE_CASES = pytest.mark.skipif(exe() is None or _model() is None or pack_path() is None,
                                  reason='no binary / model / pack (set STRATA_DSPARK_MODEL)')
# the placement the engine is measured with on this box: both cards, the tiers from the profile
BASE = ['--gpus', '0,1', '--layer-split', 'auto', '--minimax-tiers', '--minimax-prefill-batch', '4096',
        '--memory-limit-mib', '86000']


@ENGINE_CASES
def test_spec_matches_plain():
    model = _model()
    ids = _ids(int(os.environ.get('STRATA_DSPARK_PROMPT', '48')))
    n = int(os.environ.get('STRATA_DSPARK_TOKENS', '32'))
    plain, _, err0 = _run(model, ids, n, BASE)
    spec, info, err1 = _run(model, ids, n, BASE + ['--dspark-model', str(pack_path()), '--speculative',
                                                   os.environ.get('STRATA_DSPARK_SPEC', '3')])
    assert len(spec) == len(plain), (len(spec), len(plain), err1[-800:])
    assert spec == plain, f'speculation changed the tokens: {spec} != {plain}'
    assert info['spec'], err1[-800:]
    print('\nplain', plain)
    print('spec ', spec)
    print(info['spec'])


@ENGINE_CASES
def test_corrupt_drafts_still_match():
    """Every draft is wrong, so every round rolls its KV back: the tokens must not move."""
    model = _model()
    ids = _ids(int(os.environ.get('STRATA_DSPARK_PROMPT', '48')))
    n = int(os.environ.get('STRATA_DSPARK_TOKENS', '24'))
    plain, _, _ = _run(model, ids, n, BASE)
    spec, info, err = _run(model, ids, n, BASE + ['--dspark-model', str(pack_path()), '--speculative', '3'],
                           env={'STRATA_DSPARK_CORRUPT': '1'})
    assert spec == plain, f'the rollback path changed the tokens: {spec} != {plain}'
    assert 'accepted=0' in info['spec'], (info['spec'], err[-800:])


@ENGINE_CASES
def test_acceptance_and_speed():
    """Information, not a gate: the acceptance rate and both decode rates on one prompt pair."""
    model = _model()
    ids = _ids(int(os.environ.get('STRATA_DSPARK_PROMPT', '48')))
    n = int(os.environ.get('STRATA_DSPARK_TOKENS', '32'))
    for rep in range(2):
        toks, plain, _ = _run(model, ids, n, BASE)
        pspec, spec, _ = _run(model, ids, n, BASE + ['--dspark-model', str(pack_path()), '--speculative',
                                                     os.environ.get('STRATA_DSPARK_SPEC', '3')])
        same = toks == pspec
        print(f'\nrep {rep}: plain {plain["gen"] / plain["decode_ms"] * 1000:.2f} tok/s '
              f'({plain["decode_ms"]:.0f} ms for {plain["gen"]}), '
              f'spec {spec["gen"] / spec["decode_ms"] * 1000:.2f} tok/s ({spec["decode_ms"]:.0f} ms), '
              f'tokens {"identical" if same else "DIFFER"}')
        print('  ', spec['spec'])
