"""Exact dequantization for the MiniMax-M3.1-Flash release weights.

Two storage formats are used by the checkpoint (see
``_ref/MINIMAX_REFERENCE_SPEC.md`` sections A and B):

NVFP4 ("nvfp4-pack-quantized"), used by the routed MoE experts
    ``weight_packed``  U8    [O, I/2]   two E2M1 nibbles per byte,
                                        low nibble = even input index
    ``weight_scale``   F8E4M3[O, I/16]  one scale per 16 input channels
    ``weight_global_scale`` F32 scalar  a *divisor*:
                                        ``W = e2m1(nibble) * scale / global``

MXFP8, used by attention, indexer, dense MLP and shared experts
    ``weight``         F8E4M3 [O, I]
    ``weight_scale_inv`` U8   [O, I/32] UE8M0 exponent byte, *multiplied*:
                                        ``W = e4m3(w) * 2**(byte - 127)``

Everything is done in fp32 (the products are exact there: 4-bit x 4-bit
mantissas) and optionally cast to bf16 at the end.

Run ``python dequant.py --self-test`` for the hand-built unit checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

# kE2M1ToFloat: magnitude of the low three bits of an E2M1 nibble.
#   nibble = (sign << 3) | mag, value = +/- table[mag], +/-0.0 preserved.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

NVFP4_GROUP = 16
MXFP8_BLOCK = 32
E4M3_MAX = 448.0
E2M1_MAX = 6.0


def _e2m1_table(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(E2M1_MAGNITUDES, dtype=dtype)


def unpack_fp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """U8 [O, I/2] -> int8 [O, I] holding the raw nibble (sign bit still in bit 3).

    Low nibble is the *even* input index, high nibble the odd one.  This is the
    ``torch.stack((low, high), dim=1).flatten()`` order from
    compressed-tensors' ``unpack_fp4_from_uint8``.
    """
    if packed.dim() != 2:
        raise ValueError(f"packed must be 2-D [O, I/2], got {tuple(packed.shape)}")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    b = packed.to(torch.int16)
    low = b & 0x0F
    high = (b >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def nvfp4_to_float(packed: torch.Tensor) -> torch.Tensor:
    """Unpack the E2M1 values only (no block/global scale), as fp32."""
    nib = unpack_fp4_nibbles(packed)
    table = _e2m1_table(torch.float32).to(device=nib.device)
    val = table[(nib & 0x7).long()]
    # Multiply by -1 for the sign bit: keeps -0.0 for nibble 0x8, like the spec.
    return torch.where((nib & 0x8) != 0, val * -1.0, val)


def dequant_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact NVFP4 dequantization: ``e2m1(nibble) * scale / global_scale``.

    packed:       U8    [O, I/2]
    scale:        any float dtype (E4M3 in the checkpoint) [O, I/16]
    global_scale: scalar float (F32 in the checkpoint); it is a DIVISOR.
    """
    if scale.dim() != 2 or scale.shape[0] != packed.shape[0]:
        raise ValueError(f"scale shape {tuple(scale.shape)} does not match packed {tuple(packed.shape)}")
    o, half = packed.shape
    width = half * 2
    if width % NVFP4_GROUP:
        raise ValueError(f"input width {width} is not a multiple of {NVFP4_GROUP}")
    if scale.shape[1] != width // NVFP4_GROUP:
        raise ValueError(
            f"scale has {scale.shape[1]} groups per row, expected {width // NVFP4_GROUP} "
            f"(width {width}, group {NVFP4_GROUP})"
        )

    val = nvfp4_to_float(packed)
    gs = float(global_scale)
    if gs == 0.0 or not math.isfinite(gs):
        raise ValueError(f"invalid NVFP4 global scale {gs!r}")
    # s / G, computed as one fp32 multiply per scale element (spec A.4).
    s = scale.to(torch.float32) * (1.0 / gs)
    s = s.repeat_interleave(NVFP4_GROUP, dim=1)
    return (val * s).to(dtype)


def ue8m0_to_float(scale_u8: torch.Tensor) -> torch.Tensor:
    """UE8M0 exponent byte -> fp32 multiplier ``2**(byte - 127)`` (exact)."""
    if scale_u8.dtype != torch.uint8:
        raise TypeError(f"UE8M0 scale must be uint8, got {scale_u8.dtype}")
    # exp2 is exact for every integer exponent (including 2**-127, a subnormal
    # fp32).  The (v << 23) bit trick gives 0 instead of 2**-127 for v == 0, so
    # it is only equivalent for 1 <= v <= 254.
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def dequant_mxfp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact MXFP8 dequantization: ``e4m3(w) * 2**(byte - 127)`` (multiply!).

    weight:    F8E4M3 [O, I] (or any dtype torch can cast to fp32)
    scale_inv: U8      [O, ceil(I/32)]  (the ``_inv`` name is a legacy misnomer)
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [O, I], got {tuple(weight.shape)}")
    if scale_inv.dtype != torch.uint8:
        raise TypeError(f"weight_scale_inv must be uint8, got {scale_inv.dtype}")
    o, width = weight.shape
    if width % MXFP8_BLOCK:
        raise ValueError(f"input width {width} is not a multiple of {MXFP8_BLOCK}")
    nblocks = width // MXFP8_BLOCK
    if scale_inv.shape != (o, nblocks):
        raise ValueError(f"scale shape {tuple(scale_inv.shape)} != {(o, nblocks)}")
    d = ue8m0_to_float(scale_inv)  # [O, nblocks], exact powers of two
    w = weight.to(torch.float32).view(o, nblocks, MXFP8_BLOCK)
    return (w * d.unsqueeze(-1)).reshape(o, width).to(dtype)


def dequant_tensor(name: str, tensors: dict, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize checkpoint tensor ``name`` from a dict keyed by full names.

    The dict must contain the companion tensors (``weight_scale_inv`` for
    MXFP8, ``weight_scale``/``weight_global_scale`` for NVFP4).
    """
    if name.endswith("weight_packed"):
        base = name[: -len("weight_packed")]
        try:
            return dequant_nvfp4(
                tensors[base + "weight_packed"],
                tensors[base + "weight_scale"],
                tensors[base + "weight_global_scale"],
                dtype,
            )
        except KeyError as exc:
            raise KeyError(f"{name}: missing NVFP4 companion tensor {exc}") from None
    if name.endswith("weight"):
        sname = name[: -len("weight")] + "weight_scale_inv"
        if sname in tensors:
            return dequant_mxfp8(tensors[name], tensors[sname], dtype)
    # Plain (bf16/fp32) tensor, e.g. embeddings, norms, router gate.
    return tensors[name].to(dtype)


# ---------------------------------------------------------------------------
# self-test


def _pack_nvfp4_reference(values: torch.Tensor) -> torch.Tensor:
    """Pack fp32 values that are already exact E2M1 (test helper)."""
    flat = values.reshape(values.shape[0], -1)
    table = torch.tensor(E2M1_MAGNITUDES)
    out = torch.empty(flat.shape[0], flat.shape[1] // 2, dtype=torch.uint8)
    for b in range(flat.shape[1] // 2):
        nibbles = []
        for j in (2 * b, 2 * b + 1):
            v = flat[:, j]
            mag = torch.argmin((table[None, :] - v.abs()[:, None]).abs(), dim=1).to(torch.int16)
            sign = (v < 0).to(torch.int16) << 3
            nibbles.append((mag | sign).to(torch.uint8))
        out[:, b] = nibbles[0] | (nibbles[1] << 4)
    return out


def _test_nvfp4() -> None:
    # Hand-built: one row, 16 inputs = 8 bytes, one scale group.
    # nibbles: 0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7, 0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF
    # low nibble first, so byte b = nib[2b] | nib[2b+1] << 4
    nib = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF]
    packed = torch.tensor([[nib[2 * b] | (nib[2 * b + 1] << 4) for b in range(8)]], dtype=torch.uint8)
    expected_vals = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]]
    )
    got = nvfp4_to_float(packed)
    assert torch.equal(got, expected_vals), got

    # scale 1.0 in E4M3 for the single group, global divisor 2.0 -> W = val / 2
    s = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
    w = dequant_nvfp4(packed, s, torch.tensor(2.0))
    assert torch.allclose(w, expected_vals * 0.5, atol=0, rtol=0), w
    # 0x8 must stay negative zero (sign applied as a multiply)
    assert math.copysign(1.0, float(w[0, 8])) < 0

    # A real E4M3 scale 1.5 -> val * 1.5 / 2.0 ; also check exactness in bf16
    s15 = torch.tensor([[1.5]], dtype=torch.float8_e4m3fn)
    w15 = dequant_nvfp4(packed, s15, torch.tensor(2.0))
    assert torch.equal(w15, expected_vals * 0.75), w15

    # Round-trip a small random tensor through the reference packer.
    torch.manual_seed(0)
    vals = torch.tensor(E2M1_MAGNITUDES)[torch.randint(0, 8, (3, 32))] * torch.where(
        torch.rand(3, 32) < 0.5, -1.0, 1.0
    )
    p = _pack_nvfp4_reference(vals)
    s = torch.tensor([[0.5, 1.0], [2.0, 3.0], [1.0, 0.25]], dtype=torch.float8_e4m3fn)
    g = torch.tensor(3.5)
    w = dequant_nvfp4(p, s, g)
    want = vals * (s.to(torch.float32).repeat_interleave(16, 1) * (1.0 / 3.5))
    assert torch.equal(w, want)
    assert dequant_nvfp4(p, s, g, torch.bfloat16).dtype == torch.bfloat16


def _test_mxfp8() -> None:
    # hand-built 2x64
    torch.manual_seed(1)
    w = (torch.randn(2, 64) * 100).to(torch.float8_e4m3fn)
    # UE8M0 bytes: 127 -> x1, 128 -> x2, 126 -> x0.5
    s = torch.tensor([[127, 128], [126, 130]], dtype=torch.uint8)
    got = dequant_mxfp8(w, s)
    d = torch.tensor([[1.0, 2.0], [0.5, 8.0]])
    wf = w.to(torch.float32).view(2, 2, 32) * d[:, :, None]
    want = wf.reshape(2, 64)
    assert torch.equal(got, want)
    assert dequant_mxfp8(w, s, torch.bfloat16).dtype == torch.bfloat16
    assert torch.equal(ue8m0_to_float(torch.tensor([127, 128, 126, 0], dtype=torch.uint8)),
                       torch.tensor([1.0, 2.0, 0.5, 2.0 ** -127]))
    # E4M3 -> fp32 must be exact (4-bit mantissa)
    assert torch.equal(w.to(torch.float32).to(torch.float8_e4m3fn), w)


def self_test() -> None:
    _test_nvfp4()
    _test_mxfp8()
    print("dequant self-test: OK")


# ---------------------------------------------------------------------------
# stats on a real checkpoint tensor


def _stats(t: torch.Tensor) -> dict:
    x = t.to(torch.float32)
    n = x.numel()
    finite = torch.isfinite(x)
    n_nan = int((~finite).sum())
    xf = x[finite]
    return {
        "shape": list(t.shape),
        "numel": n,
        "mean": float(xf.mean()) if xf.numel() else float("nan"),
        "std": float(xf.std()) if xf.numel() else float("nan"),
        "absmax": float(xf.abs().max()) if xf.numel() else float("nan"),
        "frac_zero": float((x == 0).float().mean()),
        "frac_nan_inf": n_nan / n,
    }


def stats_cli(args) -> None:
    from safetensors import safe_open

    model_dir = Path(args.model_dir)
    index_path = Path(args.index) if args.index else model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    wanted = args.tensors
    if not wanted:
        # one representative of each quantised kind plus a plain tensor
        for pat in (
            "language_model.model.embed_tokens.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.3.block_sparse_moe.experts.0.w1.weight_packed",
            "layers.3.block_sparse_moe.experts.0.w2.weight_packed",
            "layers.3.block_sparse_moe.gate.weight",
        ):
            wanted.extend(k for k in weight_map if pat in k)
    for name in wanted:
        if name not in weight_map:
            print(f"{name}: NOT IN INDEX")
            continue
        path = model_dir / weight_map[name]
        if not path.exists():
            print(f"{name}: shard missing ({path.name})")
            continue
        base = name[: -len("weight_packed")] if name.endswith("weight_packed") else name[: -len("weight")]
        need = [name]
        if name.endswith("weight_packed"):
            need += [base + "weight_scale", base + "weight_global_scale"]
        elif name.endswith("weight"):
            need.append(base + "weight_scale_inv")
        tensors = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for k in need:
                if k in keys:
                    tensors[k] = f.get_tensor(k)
        w = dequant_tensor(name, tensors, torch.float32)
        print(name, json.dumps(_stats(w)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--model-dir", default=r"E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash")
    ap.add_argument("--index", default=r"E:\Jon's Projects\AI\Strata Fork Minimax\_ref\minimax\model.safetensors.index.json")
    ap.add_argument("--tensors", nargs="*", default=[])
    args = ap.parse_args()
    if args.self_test:
        self_test()
    if args.stats:
        stats_cli(args)
    if not (args.self_test or args.stats):
        ap.print_help()


if __name__ == "__main__":
    main()
