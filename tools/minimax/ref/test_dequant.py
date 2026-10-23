"""Unit tests for the dequantizers (hand-built tiny tensors).

Run:  python -m pytest tools/minimax/ref/test_dequant.py
or:   python tools/minimax/ref/dequant.py --self-test
"""

from __future__ import annotations

import math

import pytest
import torch

from dequant import (
    dequant_mxfp8,
    dequant_nvfp4,
    nvfp4_to_float,
    unpack_fp4_nibbles,
    ue8m0_to_float,
)

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_nibble_order_low_is_even():
    # byte = low | high<<4; low is input index 2b, high is 2b+1
    packed = torch.tensor([[0x21, 0xF0]], dtype=torch.uint8)  # (1,2) -> 4 values
    nib = unpack_fp4_nibbles(packed)
    assert nib.tolist() == [[1, 2, 0, 15]]
    vals = nvfp4_to_float(packed)
    assert vals.tolist() == [[0.5, 1.0, 0.0, -6.0]]


def test_e2m1_table_and_signed_zero():
    nib = list(range(16))
    packed = torch.tensor([[nib[2 * b] | (nib[2 * b + 1] << 4) for b in range(8)]], dtype=torch.uint8)
    got = nvfp4_to_float(packed)
    want = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]])
    assert torch.equal(got, want)
    assert math.copysign(1.0, float(got[0, 8])) < 0  # -0.0 preserved


def test_nvfp4_global_scale_is_a_divisor():
    packed = torch.full((1, 8), 0x11, dtype=torch.uint8)  # 16 values of 0.5
    s = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
    assert dequant_nvfp4(packed, s, torch.tensor(4.0)).tolist() == [[0.25] * 16]
    assert dequant_nvfp4(packed, s, torch.tensor(1.0)).tolist() == [[1.0] * 16]


def test_nvfp4_group_of_16():
    torch.manual_seed(0)
    # 2 rows, 32 inputs -> 2 groups per row, 16 packed bytes
    vals = torch.tensor(E2M1)[torch.randint(0, 8, (2, 32))] * torch.where(torch.rand(2, 32) < 0.5, -1.0, 1.0)
    table = torch.tensor(E2M1)
    mag = torch.argmin((table[None, None, :] - vals.abs()[:, :, None]).abs(), -1)
    nib = mag + (vals < 0).to(torch.long) * 8
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    s = torch.tensor([[0.5, 2.0], [1.0, 4.0]], dtype=torch.float8_e4m3fn)
    got = dequant_nvfp4(packed, s, torch.tensor(2.0))
    want = vals * (s.to(torch.float32).repeat_interleave(16, 1) * 0.5)
    assert torch.equal(got, want)


def test_nvfp4_rejects_bad_shapes():
    packed = torch.zeros(1, 4, dtype=torch.uint8)
    with pytest.raises(ValueError):
        dequant_nvfp4(packed, torch.zeros(1, 1, dtype=torch.float8_e4m3fn), 1.0)
    with pytest.raises(ValueError):
        dequant_nvfp4(packed, torch.zeros(1, 1, dtype=torch.float8_e4m3fn), 0.0)


def test_ue8m0_exponent():
    got = ue8m0_to_float(torch.tensor([127, 128, 126, 0], dtype=torch.uint8))
    assert torch.equal(got, torch.tensor([1.0, 2.0, 0.5, 2.0 ** -127]))


def test_mxfp8_multiplies_block_scales():
    torch.manual_seed(1)
    w = (torch.randn(2, 64) * 100).to(torch.float8_e4m3fn)
    s = torch.tensor([[127, 128], [126, 130]], dtype=torch.uint8)
    got = dequant_mxfp8(w, s)
    d = torch.tensor([[1.0, 2.0], [0.5, 8.0]])
    want = (w.to(torch.float32).view(2, 2, 32) * d[:, :, None]).reshape(2, 64)
    assert torch.equal(got, want)
    # dividing by the scale instead would be wrong by 2**(2*(b-127))
    assert not torch.equal(got, w.to(torch.float32) / d.repeat_interleave(32, 1))


def test_mxfp8_e4m3_to_fp32_is_exact():
    torch.manual_seed(2)
    w = (torch.randn(4, 32) * 400).to(torch.float8_e4m3fn)
    back = w.to(torch.float32).to(torch.float8_e4m3fn)
    assert torch.equal(w, back)


def test_dtypes():
    packed = torch.tensor([[0x11] * 8], dtype=torch.uint8)
    s = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
    assert dequant_nvfp4(packed, s, 1.0, torch.bfloat16).dtype == torch.bfloat16
    w = torch.zeros(1, 32, dtype=torch.float8_e4m3fn)
    assert dequant_mxfp8(w, torch.full((1, 1), 127, dtype=torch.uint8), torch.bfloat16).dtype == torch.bfloat16
