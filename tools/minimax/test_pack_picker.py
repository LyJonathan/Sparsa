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
    (64, None),          # no pack fits; the fallback cannot cover D3 in RAM+VRAM either
    (96, None),          # IQ1_S fits but is stale and below the 2.0 tok/s floor; D3 needs 132.3 > 96+24
    (128, "iq2-xs-d3"),  # fallback: 132.3 <= 128+24, streamed from SSD at the measured 4.0 tok/s
    (192, "iq2-xs-d3"),  # now it fits RAM, so it is a normal pick
    (256, "iq2-xs-d3"),  # no faster floor-clearing pack exists
])
def test_pick_by_ram_and_gpus(ram, gpus, want):
    machine = sparsa.Machine(ram_gb=ram, vram_gb=gpus, free_disk_gb=2000)
    got = sparsa.choose_pack(machine)
    assert (got["key"] if got else None) == want


def test_128_gb_streams_d3_from_ssd_and_says_so():
    got, reason = sparsa.select_pack(sparsa.Machine(128, (12.0, 12.0), 2000))
    assert got["key"] == "iq2-xs-d3"
    assert reason == sparsa.STREAMING_REASON == "experts don't fit in RAM; streaming from SSD"


def test_96_gb_does_not_pick_iq1_s_below_the_floor():
    got, reason = sparsa.select_pack(sparsa.Machine(96, (24.0,), 2000))
    assert got is None and reason == ""
    iq1s = next(p for p in sparsa.PACKS if p["key"] == "iq1-s")
    assert iq1s["decode_tok_s"] < sparsa.DEFAULT_SPEED_FLOOR_TOK_S
    assert iq1s["speed_stale"]


def test_iq2_kt_is_selectable_but_never_a_default():
    got, _ = sparsa.select_pack(sparsa.Machine(128, (12.0, 12.0), 2000))
    assert got["key"] != "iq2-kt"
    kt = next(p for p in sparsa.PACKS if p["key"] == "iq2-kt")
    assert kt["decode_tok_s"] < sparsa.DEFAULT_SPEED_FLOOR_TOK_S
    assert kt["slice_decode_tok_s"] == 1.61
    xs = next(p for p in sparsa.PACKS if p["key"] == "iq2-xs")
    assert xs["slice_decode_tok_s"] == 15.99


def test_speed_provenance_and_staleness_are_recorded():
    for p in sparsa.PACKS:
        if p.get("decode_tok_s") is None:
            continue
        for key in ("speed_engine", "speed_commit", "speed_ctx", "speed_shared", "speed_stale"):
            assert key in p, (p["key"], key)
    d3 = next(p for p in sparsa.PACKS if p["key"] == "iq2-xs-d3")
    assert d3["decode_tok_s"] == 4.0 and d3["speed_stale"] is False
    assert d3["speed_engine"] == "v0.1.2"
    assert "84 GiB" in d3["speed_settings"] and "4096-token prefill chunk" in d3["speed_settings"]
    kt = next(p for p in sparsa.PACKS if p["key"] == "iq2-kt")
    assert kt["speed_stale"] is True          # G3 smoke setup, not the default setup


def test_iq2_xs_ranks_just_below_iq2_xs_d3_but_has_no_decode_speed():
    by_key = {p["key"]: p for p in sparsa.PACKS}
    assert by_key["iq2-xs"]["top1"] is None
    assert sparsa.pack_quality(by_key["iq2-xs"]) < sparsa.pack_quality(by_key["iq2-xs-d3"])
    assert sparsa.pack_quality(by_key["iq2-xs"]) > sparsa.pack_quality(by_key["iq2-kt"])
    assert by_key["iq2-xs"]["speed_stale"] is True


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
    # 150 GB of disk fits IQ2_XS-D3 (146.1 + 2 GB of mmproj/tokenizer), not IQ2_KL
    machine = sparsa.Machine(ram_gb=256, vram_gb=(24.0,), free_disk_gb=150)
    assert sparsa.choose_pack(machine)["key"] == "iq2-xs-d3"


def test_unpublished_placeholders_are_never_picked():
    machine = sparsa.Machine(ram_gb=1024, vram_gb=(80.0,) * 8, free_disk_gb=4000)
    got = sparsa.choose_pack(machine)
    assert got["key"] == "iq2-xs-d3"
    placeholders = [p for p in sparsa.PACKS if not p["available"]]
    assert placeholders and all(p["key"] != got["key"] for p in placeholders)


def test_a_published_a1_mix_with_measured_numbers_wins_at_128_gb():
    a1 = {"key": "a1-mix", "folder": "A1", "experts_gb": 100.0, "total_gb": 115.0,
          "top1": 60.5, "decode_tok_s": 3.0, "measured_on": "test", "recommended": True,
          "available": True, "about": "published A1 mix", "speed_engine": "v0.1.48",
          "speed_commit": "test", "speed_ctx": 32768, "speed_shared": False, "speed_stale": False}
    packs = (*sparsa.PACKS, a1)
    got = sparsa.choose_pack(sparsa.Machine(128, (12.0, 12.0), 2000), packs=packs)
    assert got["key"] == "a1-mix"


def test_pick_prints_the_streaming_reason(monkeypatch):
    import contextlib
    import io
    import types

    monkeypatch.setattr(sparsa, "machine_specs", lambda path=None: sparsa.Machine(128, (12.0, 12.0), 2000))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert sparsa.cmd_pick(types.SimpleNamespace(models_dir=None, json=False, context=32768,
                                                    vram_slots=-1)) == 0
    text = out.getvalue()
    assert "iq2-xs-d3" in text and sparsa.STREAMING_REASON in text
    assert "STALE" in text          # the G3 numbers are marked
