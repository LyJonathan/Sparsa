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
    (96, "iq1-s"),       # 83.9 GB of experts <= 86 GB
    (128, "iq2-xxs"),    # 110.8 <= 118; IQ2_XS's 119.4 just misses
    (192, "iq3-k"),      # 177.5 <= 182
    (256, "nvfp4"),      # 232.4 <= 246
])
def test_pick_by_ram_and_gpus(ram, gpus, want):
    machine = sparsa.Machine(ram_gb=ram, vram_gb=gpus, free_disk_gb=2000)
    got = sparsa.choose_pack(machine)
    assert (got["key"] if got else None) == want


def test_the_free_disk_is_part_of_the_fit():
    machine = sparsa.Machine(ram_gb=256, vram_gb=(24.0,), free_disk_gb=150)
    assert sparsa.choose_pack(machine)["key"] == "iq2-xs-d3"     # 146.1 + 2 GB of mmproj/tokenizer fits


def test_unpublished_placeholders_are_never_picked():
    machine = sparsa.Machine(ram_gb=1024, vram_gb=(80.0,) * 8, free_disk_gb=4000)
    assert sparsa.choose_pack(machine)["key"] == "nvfp4"
    placeholders = [p for p in sparsa.PACKS if not p["available"]]
    assert placeholders and all(p["key"] not in {sparsa.choose_pack(machine)["key"]} for p in placeholders)


def test_every_published_pack_can_be_picked_by_a_big_enough_machine():
    for pack in sparsa.PACKS:
        if not pack["available"]:
            continue
        machine = sparsa.Machine(ram_gb=pack["experts_gb"] + sparsa.RAM_RESERVE_GB,
                                 vram_gb=(24.0,), free_disk_gb=pack["total_gb"] + sparsa.DISK_RESERVE_GB)
        got = sparsa.choose_pack(machine)
        assert got is not None and got["experts_gb"] >= pack["experts_gb"] or \
            got["key"] == pack["key"]
