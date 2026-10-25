"""The pack picker: fake machine specs only, no engine, GPU or model download."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sparsa  # noqa: E402


@pytest.mark.parametrize("gpus", [(12.0,), (12.0, 12.0)])
@pytest.mark.parametrize("ram,want", [
    (64, None),          # no published pack has <= 54 GB of experts
    (96, "iq1-s"),       # only 83.9 GB of experts fits in 86 GB; not recommended, warn
    (128, "iq2-kt"),     # 56.3 % top-1 beats IQ2_XXS's 52.0; both fit
    (192, "iq3-k"),      # 177.5 <= 182, 69.8 % top-1
    (256, "nvfp4"),      # 232.4 <= 246, 77.3 % top-1
])
def test_pick_by_ram_and_gpus(ram, gpus, want):
    machine = sparsa.Machine(ram_gb=ram, vram_gb=gpus, free_disk_gb=2000)
    got = sparsa.choose_pack(machine)
    assert (got["key"] if got else None) == want


def test_iq2_kt_wins_on_quality_at_128_gb_and_iq1_s_warns_at_96_gb():
    got = sparsa.choose_pack(sparsa.Machine(128, (12.0, 12.0), 2000))
    assert got["key"] == "iq2-kt" and got["top1"] == 56.3
    assert sparsa.pack_warning(got) == ""

    got = sparsa.choose_pack(sparsa.Machine(96, (12.0,), 2000))
    assert got["key"] == "iq1-s"
    assert "WARNING" in sparsa.pack_warning(got) and "not recommended" in sparsa.pack_warning(got)


def test_iq2_xs_ranks_just_below_iq2_xs_d3():
    by_key = {p["key"]: p for p in sparsa.PACKS}
    assert by_key["iq2-xs"]["top1"] is None
    assert sparsa.pack_quality(by_key["iq2-xs"]) < sparsa.pack_quality(by_key["iq2-xs-d3"])
    assert sparsa.pack_quality(by_key["iq2-xs"]) > sparsa.pack_quality(by_key["iq2-kt"])


def test_the_free_disk_is_part_of_the_fit():
    machine = sparsa.Machine(ram_gb=256, vram_gb=(24.0,), free_disk_gb=150)
    assert sparsa.choose_pack(machine)["key"] == "iq2-xs-d3"     # 146.1 + 2 GB of mmproj/tokenizer fits


def test_unpublished_placeholders_are_never_picked():
    machine = sparsa.Machine(ram_gb=1024, vram_gb=(80.0,) * 8, free_disk_gb=4000)
    assert sparsa.choose_pack(machine)["key"] == "nvfp4"
    placeholders = [p for p in sparsa.PACKS if not p["available"]]
    assert placeholders and all(p["key"] not in {sparsa.choose_pack(machine)["key"]} for p in placeholders)


def test_every_pick_fits_the_machine_that_chose_it():
    for pack in sparsa.PACKS:
        if not pack["available"]:
            continue
        machine = sparsa.Machine(ram_gb=pack["experts_gb"] + sparsa.RAM_RESERVE_GB,
                                 vram_gb=(24.0,), free_disk_gb=pack["total_gb"] + sparsa.DISK_RESERVE_GB)
        got = sparsa.choose_pack(machine)
        assert got is not None and got["available"]
        assert got["experts_gb"] <= machine.ram_gb - sparsa.RAM_RESERVE_GB
        assert got["total_gb"] + sparsa.DISK_RESERVE_GB <= machine.free_disk_gb


def test_pick_prints_the_iq1_s_warning(monkeypatch):
    import contextlib
    import io
    import types

    monkeypatch.setattr(sparsa, "machine_specs", lambda path=None: sparsa.Machine(96, (12.0,), 2000))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert sparsa.cmd_pick(types.SimpleNamespace(models_dir=None, json=False)) == 0
    text = out.getvalue()
    assert "iq1-s" in text and "WARNING" in text and "not recommended" in text
