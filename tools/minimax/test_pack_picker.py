"""The pack picker: fake machine specs only, no engine, GPU or model download."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sparsa  # noqa: E402


@pytest.mark.parametrize("gpus", [(24.0,), (12.0, 12.0)])
@pytest.mark.parametrize("ram,want", [
    (64, None),          # the RAM tier cannot hold even IQ1_S after dense/KV/workspace and the 10 GB floor
    (96, "iq1-s"),       # only IQ1_S fits; not recommended, warn
    (128, "iq2-kt"),     # 56.3 % top-1 beats IQ2_XXS's 52.0; both fit
    (192, "iq2-kl"),     # 59.8 % measured top-1, just above IQ2_XS-D3's 59.4
    (256, "nvfp4"),      # 77.3 % top-1
])
def test_pick_by_ram_and_gpus(ram, gpus, want):
    machine = sparsa.Machine(ram_gb=ram, vram_gb=gpus, free_disk_gb=2000)
    got = sparsa.choose_pack(machine)
    assert (got["key"] if got else None) == want


def test_iq2_kt_wins_on_quality_at_128_gb_and_iq1_s_warns_at_96_gb():
    got = sparsa.choose_pack(sparsa.Machine(128, (12.0, 12.0), 2000))
    assert got["key"] == "iq2-kt" and got["top1"] == 56.3
    assert sparsa.pack_warning(got) == ""

    got = sparsa.choose_pack(sparsa.Machine(96, (24.0,), 2000))
    assert got["key"] == "iq1-s"
    assert "WARNING" in sparsa.pack_warning(got) and "not recommended" in sparsa.pack_warning(got)


def test_iq2_xs_ranks_just_below_iq2_xs_d3_but_has_no_decode_speed():
    by_key = {p["key"]: p for p in sparsa.PACKS}
    assert by_key["iq2-xs"]["top1"] is None
    assert sparsa.pack_quality(by_key["iq2-xs"]) < sparsa.pack_quality(by_key["iq2-xs-d3"])
    assert sparsa.pack_quality(by_key["iq2-xs"]) > sparsa.pack_quality(by_key["iq2-kt"])
    # it fits at 192 GB, but without a measured decode speed it cannot be the default
    assert by_key["iq2-xs"]["decode_tok_s"] is None
    assert sparsa.choose_pack(sparsa.Machine(192, (12.0, 12.0), 2000))["key"] == "iq2-kl"


def test_the_ram_tier_formula_counts_dense_kv_and_workspace():
    pack = next(p for p in sparsa.PACKS if p["key"] == "iq2-kt")
    d = sparsa.fit_detail(sparsa.Machine(128, (12.0, 12.0), 2000), pack)
    # 128 - 10 floor - ~4.7 host dense - workspace; KV is in VRAM
    assert 110 < d["ram_budget_gb"] < 116
    assert d["required_experts_gb"] < pack["experts_gb"]       # VRAM expert slots hold some
    assert d["host_kv_gb"] == 0

    small = sparsa.fit_detail(sparsa.Machine(128, (), 2000), pack)
    assert small["host_kv_gb"] > 4 and small["ram_budget_gb"] < d["ram_budget_gb"]


def test_the_free_disk_is_only_a_download_check():
    # 150 GB of disk fits IQ2_XS-D3 (146.1 + 2 GB of mmproj/tokenizer), not IQ2_KL (152.7 + 2)
    machine = sparsa.Machine(ram_gb=256, vram_gb=(24.0,), free_disk_gb=150)
    assert sparsa.choose_pack(machine)["key"] == "iq2-xs-d3"


def test_unpublished_placeholders_are_never_picked():
    machine = sparsa.Machine(ram_gb=1024, vram_gb=(80.0,) * 8, free_disk_gb=4000)
    assert sparsa.choose_pack(machine)["key"] == "nvfp4"
    placeholders = [p for p in sparsa.PACKS if not p["available"]]
    assert placeholders and all(p["key"] not in {sparsa.choose_pack(machine)["key"]} for p in placeholders)


def test_a_published_a1_mix_with_measured_numbers_wins_at_128_gb():
    a1 = {"key": "a1-mix", "folder": "A1", "experts_gb": 100.0, "total_gb": 115.0,
          "top1": 60.5, "decode_tok_s": 0.5, "measured_on": "test", "recommended": True,
          "available": True, "about": "published A1 mix"}
    packs = (*sparsa.PACKS, a1)
    got = sparsa.choose_pack(sparsa.Machine(128, (12.0, 12.0), 2000), packs=packs)
    assert got["key"] == "a1-mix"


def test_pick_prints_the_iq1_s_warning(monkeypatch):
    import contextlib
    import io
    import types

    monkeypatch.setattr(sparsa, "machine_specs", lambda path=None: sparsa.Machine(96, (24.0,), 2000))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert sparsa.cmd_pick(types.SimpleNamespace(models_dir=None, json=False, context=32768,
                                                    vram_slots=-1)) == 0
    text = out.getvalue()
    assert "iq1-s" in text and "WARNING" in text and "not recommended" in text
