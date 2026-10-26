"""sparsa.py - run MiniMax-M3.1-Flash on this PC.

Sparsa is a fork of Strata that adds a MiniMax-M3.1-Flash engine, ``strata-minimax``.  Strata's own
``strata`` command knows the Qwen families; this small program writes the run config a MiniMax model
needs and hands it to the same Python server (``serve/server.py``: OpenAI ``/v1/chat/completions``,
Anthropic ``/v1/messages``, streaming, tool calls, the web page).

    python sparsa.py serve --model D:\\models\\MiniMax-M3.1-Flash-IQ2_XS-D3
    python sparsa.py serve --model model.gguf --context 8192 --gpu 0
    python sparsa.py plan  --model model.gguf          # print the settings, start nothing
    python sparsa.py download iq2-xs-d3 --dir D:\\models
    python sparsa.py download mmproj --dir D:\\models\\MiniMax-M3.1-Flash-IQ2_XS-D3
    python sparsa.py inspect --model model.gguf        # what the engine reads from the file

The model argument is the folder the GGUF parts were downloaded into, or the first part.  The
tokenizer is needed too (``tokenizer.json`` from the checkpoint, ~10 MB); ``download tokenizer``
fetches it, and ``serve`` says where it looked when it cannot find it.

Vision is one more file: ``download mmproj`` puts a ~1.7 GB ``mmproj-*.gguf`` beside the parts, and
``serve`` finds it there (``--mmproj PATH`` names another one, ``--no-vision`` ignores it).  Without
it the server is text only.

EXPERIMENTAL.  Decode is slow (a few tokens a second on two consumer cards), long context is work
in progress, and pictures go through a CPU fp32 encoder: 64-80 s each.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE_NAME = "strata-minimax.exe" if os.name == "nt" else "strata-minimax"
MODEL_NAME = "minimax-m3.1-flash"
EXPECTED_PARTS = 13
GGUF_PART = "-00001-of-"

# LyJonathan/MiniMax-M3.1-Flash-GGUF: the published tiers, plus the planned mixed tiers as placeholders.
# `experts_gb` is the sum of the routed `*_exps` tensors, read from each pack's GGUF headers; `total_gb`
# is the whole 13-part download (the published README sizes). `top1` is the Sparsa G3 result from the HF
# README Quality table. `decode_tok_s` and `measured_on` are measured full-model decode numbers; a pack
# without one is shown but is never auto-picked. `default_ok` is false for the not-recommended IQ1_S and
# for placeholders. The picker only considers `available` packs.
PC = "2x RTX 5070 12 GB, Ryzen 9 5900X, 128 GB"
CLOUD = "RTX PRO 6000 96 GB, 160 GiB RAM"
PACKS = (
    {"key": "iq1-s", "folder": "MiniMax-M3.1-Flash-IQ1_S", "experts_gb": 83.915, "total_gb": 94.786,
     "top1": 44.3, "decode_tok_s": 1.85, "measured_on": PC, "recommended": False, "available": True,
     "about": "IQ1_S experts; 44.3 % top-1; NOT RECOMMENDED (it loops in practice)"},
    {"key": "iq1-m", "folder": "MiniMax-M3.1-Flash-IQ1_M", "experts_gb": 95.750, "total_gb": 106.620,
     "top1": 47.1, "decode_tok_s": 0.90, "measured_on": CLOUD, "recommended": True, "available": True,
     "about": "IQ1_M experts; 47.1 % top-1"},
    {"key": "iq2-kt", "folder": "MiniMax-M3.1-Flash-IQ2_KT", "experts_gb": 110.094, "total_gb": 123.820,
     "top1": 56.3, "decode_tok_s": 0.35, "measured_on": PC, "recommended": True, "available": True,
     "slice_decode_tok_s": 1.61, "slice_measured_on": "15-layer slice, one RTX 5070, Ryzen 9 5900X, 128 GB",
     "about": "IQ2_KT experts (ik_llama.cpp's type); Sparsa only; 56.3 % top-1"},
    {"key": "iq2-xxs", "folder": "MiniMax-M3.1-Flash-IQ2_XXS", "experts_gb": 110.811, "total_gb": 124.537,
     "top1": 52.0, "decode_tok_s": 0.66, "measured_on": PC, "recommended": True, "available": True,
     "about": "IQ2_XXS experts; 52.0 % top-1"},
    {"key": "iq2-xs", "folder": "MiniMax-M3.1-Flash-IQ2_XS", "experts_gb": 119.418, "total_gb": 133.143,
     "top1": None, "quality_rank": 59.3, "decode_tok_s": None, "measured_on": "not measured",
     "recommended": True, "available": True, "slice_decode_tok_s": 15.99,
     "slice_measured_on": "15-layer slice, one RTX 5070, Ryzen 9 5900X, 128 GB",
     "about": "IQ2_XS experts; no Sparsa G3 number (ranked just below IQ2_XS-D3); no full-model decode speed"},
    {"key": "iq2-xs-d3", "folder": "MiniMax-M3.1-Flash-IQ2_XS-D3", "experts_gb": 132.328, "total_gb": 146.054,
     "top1": 59.4, "decode_tok_s": 4.0, "measured_on": PC + " (HF README)", "recommended": True,
     "available": True, "about": "IQ2_XS gate/up + IQ3_XXS down; 59.4 % top-1"},
    {"key": "iq2-kl", "folder": "MiniMax-M3.1-Flash-IQ2_KL", "experts_gb": 138.963, "total_gb": 152.688,
     "top1": 59.8, "decode_tok_s": 0.27, "measured_on": CLOUD, "recommended": True, "available": True,
     "about": "IQ2_KL experts; Sparsa only; 59.8 % top-1"},
    {"key": "iq3-k", "folder": "MiniMax-M3.1-Flash-IQ3_K", "experts_gb": 177.513, "total_gb": 191.239,
     "top1": 69.8, "decode_tok_s": None, "measured_on": "not measured", "recommended": True,
     "available": True, "about": "IQ3_K experts; Sparsa only; 69.8 % top-1"},
    {"key": "iq4-ks", "folder": "MiniMax-M3.1-Flash-IQ4_KS", "experts_gb": 219.830, "total_gb": 233.555,
     "top1": 74.4, "decode_tok_s": None, "measured_on": "not measured", "recommended": True,
     "available": True, "about": "IQ4_KS experts; Sparsa only; 74.4 % top-1"},
    {"key": "nvfp4", "folder": "MiniMax-M3.1-Flash-NVFP4", "experts_gb": 232.381, "total_gb": 246.107,
     "top1": 77.3, "decode_tok_s": 0.14, "measured_on": CLOUD, "recommended": True, "available": True,
     "about": "the checkpoint's NVFP4 expert values; Sparsa only; 77.3 % top-1"},
    # Planned, not published: a mixed pack with about 100 GB of experts for a 128 GB machine, and an
    # IQ1_S-based pack for a 96 GB machine. When the mixed pack's measured top-1, decode and size are
    # filled in and `available` becomes true, it competes normally and wins when it is the best fit.
    {"key": "mix-128", "folder": "MiniMax-M3.1-Flash-MIX128", "experts_gb": 100.0, "total_gb": 0.0,
     "top1": None, "decode_tok_s": None, "measured_on": "not published", "recommended": True,
     "available": False, "about": "planned mixed ~100 GB-expert pack for 128 GB RAM; not published yet"},
    {"key": "iq1-s-96", "folder": "MiniMax-M3.1-Flash-IQ1_S-96", "experts_gb": 84.0, "total_gb": 0.0,
     "top1": None, "decode_tok_s": None, "measured_on": "not published", "recommended": True,
     "available": False, "about": "planned IQ1_S-based pack for 96 GB RAM; not published yet"},
)
# Speed provenance. A number is stale when it was not measured on engine v0.1.3 or later with the default
# setup; a stale number cannot satisfy DEFAULT_SPEED_FLOOR_TOK_S and cannot make a pack the default.
SPEED_META = {
    "iq1-s": {"speed_engine": "v0.1.48", "speed_commit": "unrecorded (G3 smoke)", "speed_ctx": 8192,
              "speed_shared": False, "speed_stale": True,
              "speed_note": "G3 smoke flags, not the default setup"},
    "iq1-m": {"speed_engine": "v0.1.48", "speed_commit": "unrecorded (G3 cloud smoke)", "speed_ctx": 8192,
              "speed_shared": False, "speed_stale": True, "speed_note": "cloud G3 smoke, not the default PC setup"},
    "iq2-kt": {"speed_engine": "v0.1.48", "speed_commit": "unrecorded (G3 smoke)", "speed_ctx": 8192,
               "speed_shared": False, "speed_stale": True, "speed_note": "G3 smoke flags, not the default setup",
               "slice_engine": "v0.1.48", "slice_commit": "unrecorded", "slice_ctx": 1024,
               "slice_shared": False, "slice_stale": False},
    "iq2-xxs": {"speed_engine": "v0.1.48", "speed_commit": "unrecorded (G3 smoke)", "speed_ctx": 8192,
                "speed_shared": False, "speed_stale": True, "speed_note": "G3 smoke flags, not the default setup"},
    "iq2-xs": {"speed_engine": "not measured", "speed_commit": "-", "speed_ctx": 0,
               "speed_shared": False, "speed_stale": True, "speed_note": "no full-model decode number",
               "slice_engine": "v0.1.48", "slice_commit": "unrecorded", "slice_ctx": 1024,
               "slice_shared": False, "slice_stale": False},
    # D3's number is the reference baseline. A future full-model speed must match these settings or be
    # marked not comparable (speed_stale=True / its own note).
    "iq2-xs-d3": {"speed_engine": "v0.1.2", "speed_commit": "v0.1.2 release measurement, repeated as ~4 tok/s in v0.1.3 README",
                  "speed_ctx": 32768, "speed_shared": False, "speed_stale": False,
                  "speed_settings": "sparsa.py serve defaults: 4096-token prefill chunk, int8 expert dots, "
                                    "RAM tier capped at 84 GiB, 64 greedy tokens, idle box",
                  "speed_note": "reference baseline for comparable speeds"},
    "iq2-kl": {"speed_engine": "v0.1.48", "speed_commit": "unrecorded (G3 cloud smoke)", "speed_ctx": 8192,
               "speed_shared": False, "speed_stale": True, "speed_note": "cloud G3 smoke, not the default PC setup"},
    "iq3-k": {"speed_engine": "not measured", "speed_commit": "-", "speed_ctx": 0,
              "speed_shared": False, "speed_stale": True, "speed_note": "no measured decode number"},
    "iq4-ks": {"speed_engine": "not measured", "speed_commit": "-", "speed_ctx": 0,
               "speed_shared": False, "speed_stale": True, "speed_note": "no measured decode number"},
    "nvfp4": {"speed_engine": "v0.1.48", "speed_commit": "unrecorded (G3 cloud smoke)", "speed_ctx": 8192,
              "speed_shared": False, "speed_stale": True, "speed_note": "cloud G3 smoke, not the default PC setup"},
}
for _pack in PACKS:
    _pack.update(SPEED_META.get(_pack["key"], {"speed_engine": "not published", "speed_commit": "-",
                                               "speed_ctx": 0, "speed_shared": False, "speed_stale": True,
                                               "speed_note": "not published"}))
QUANTS = {p["key"]: (p["folder"], p["about"]) for p in PACKS}

# The rule: best measured quality (Sparsa G3 top-1) among packs whose experts fit in the engine's RAM
# expert tier and whose download fits the disk. Ties go to the smaller pack.
#
# The fit arithmetic follows src/minimax/engine.cpp's "minimax-ram-tier plan" line and
# include/strata/minimax/vram_plan.hpp. The engine's own headroom is 1/16 of physical RAM clamped to
# 3..10 GiB; the requirement here is the 10 GB floor, so the larger of the two is the same 10 GiB.
RAM_FLOOR_GB = 10.0
DISK_RESERVE_GB = 2.0
DEFAULT_CONTEXT = 32768
# A pack can be an auto-pick only with a fresh measured decode at or above this floor.
DEFAULT_SPEED_FLOOR_TOK_S = 2.0

# MiniMax-M3.1-Flash geometry (config.json) and the engine's host buffers.
MODEL_LAYERS = 60
MOE_LAYERS = 57
N_EMBD = 6144
N_FF_EXP = 3072
N_VOCAB = 200064
HEAD_DIM = 128
N_HEAD_KV = 4
IDX_DIM = 128
# 135 KiB per token, the same figure pc_launch.py and the engine use for host K/V.
KV_GB_PER_TOKEN = 135.0 / 1024.0 / 1024.0
VRAM_RESERVE_GB = 700.0 / 1024.0          # --vram-reserve-mib default
PREFILL_WORKSPACE_GB = 0.55              # the 4096-token batched-prompt workspace per card (engine log)

# The dense weights that stay host-side even when every layer is on a GPU: shared experts + embeddings.
SHARED_EXPERT_PARAMS = MOE_LAYERS * 3 * N_EMBD * N_FF_EXP
EMBED_PARAMS = N_VOCAB * N_EMBD
ATTENTION_PARAMS = 6.65e9
DENSE_MLP_PARAMS = 0.68e9
OUTPUT_PARAMS = N_VOCAB * N_EMBD
DENSE_PARAMS = SHARED_EXPERT_PARAMS + EMBED_PARAMS + ATTENTION_PARAMS + DENSE_MLP_PARAMS + OUTPUT_PARAMS
HOST_DENSE_FRACTION = (SHARED_EXPERT_PARAMS + EMBED_PARAMS) / DENSE_PARAMS
HOST_WORK_GB = (N_VOCAB + 8 * N_FF_EXP + 16 * N_EMBD + 4096) * 4 / 1e9


@dataclass(frozen=True)
class Machine:
    """What one machine offers, in the numbers people call them (a 128 GB machine reports 128.0)."""
    ram_gb: float
    vram_gb: tuple[float, ...]
    free_disk_gb: float

    @property
    def gpu_count(self) -> int:
        return len(self.vram_gb)


def total_ram_gb() -> float:
    """Installed RAM as the number people call it (128 GB -> 128.0)."""
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / (1 << 30)
    else:
        try:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1 << 30)
        except (ValueError, OSError):
            pass
    return 0.0


def gpu_vram_gb() -> tuple[float, ...]:
    """Each NVIDIA card's total memory in GB, in nvidia-smi's order; empty without a card."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return ()
    if out.returncode != 0:
        return ()
    cards = []
    for line in out.stdout.splitlines():
        line = line.strip().split()[0] if line.strip() else ""
        try:
            cards.append(float(line) / 1024.0)       # MiB -> GB
        except ValueError:
            continue
    return tuple(cards)


def free_disk_gb(path: str | os.PathLike | None = None) -> float:
    """Free space on the volume `path` is (or will be) on; a folder that does not exist yet uses its parent."""
    probe = Path(path or Path.cwd()).expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / (1 << 30)
    except OSError:
        return 0.0


def machine_specs(path: str | os.PathLike | None = None) -> Machine:
    return Machine(total_ram_gb(), gpu_vram_gb(), free_disk_gb(path))


def pack_quality(pack: dict) -> float:
    """The pack's Sparsa G3 top-1 for ranking; IQ2_XS has no G3 run and uses its explicit rank just below
    IQ2_XS-D3. A placeholder with no number ranks last (and is never selectable anyway)."""
    return pack.get("quality_rank", pack.get("top1") if pack.get("top1") is not None else -1.0)


def fit_detail(machine: Machine, pack: dict, ctx: int = DEFAULT_CONTEXT, vram_slots: int = -1) -> dict:
    """The RAM expert budget and what this pack needs, from the engine's own constants.

    `budget = min(physical, available) - 10 GiB - host dense - host K/V - host workspace` (the engine's
    ram-tier plan with the required 10 GB floor). VRAM holds every dense layer that fits after K/V and
    the reserve; any free VRAM then holds expert slots, which reduce what the RAM tier must hold.
    `vram_slots`: -1/auto, 0/off, or a per-layer count.
    """
    dense_gb = max(0.0, pack["total_gb"] - pack["experts_gb"])
    host_dense = dense_gb
    dense_vram = 0.0
    gpu_layers = 0
    host_kv = ctx * KV_GB_PER_TOKEN
    if machine.vram_gb:
        reserve = VRAM_RESERVE_GB * machine.gpu_count
        kv_vram = ctx * KV_GB_PER_TOKEN
        per_layer = dense_gb * (1.0 - HOST_DENSE_FRACTION) / MODEL_LAYERS
        usable = sum(machine.vram_gb) - reserve - kv_vram
        gpu_layers = 0 if per_layer <= 0 else max(0, min(MODEL_LAYERS, int(usable / per_layer)))
        dense_vram = min(dense_gb * (1.0 - HOST_DENSE_FRACTION), gpu_layers * per_layer)
        host_dense = max(0.0, dense_gb - dense_vram)
        host_kv = 0.0                                    # sparsa passes --minimax-kv vram

    budget = machine.ram_gb - RAM_FLOOR_GB - host_dense - host_kv - HOST_WORK_GB
    vram_experts = 0.0
    if machine.vram_gb and vram_slots != 0:
        free = (sum(machine.vram_gb) - VRAM_RESERVE_GB * machine.gpu_count
                - ctx * KV_GB_PER_TOKEN - dense_vram - PREFILL_WORKSPACE_GB * machine.gpu_count)
        per_expert_layer = pack["experts_gb"] / MOE_LAYERS      # one slot on every MoE layer
        slots = max(0, int(free / per_expert_layer)) if per_expert_layer > 0 else 0
        if vram_slots > 0:
            slots = min(slots, vram_slots)
        slots = min(slots, 128)
        vram_experts = min(pack["experts_gb"], slots * per_expert_layer)
    required = max(0.0, pack["experts_gb"] - vram_experts)
    return {"ram_budget_gb": max(0.0, budget), "required_experts_gb": required,
            "vram_experts_gb": vram_experts, "host_dense_gb": host_dense,
            "host_kv_gb": host_kv, "gpu_layers": gpu_layers}


def pack_fits(machine: Machine, pack: dict, ctx: int = DEFAULT_CONTEXT, vram_slots: int = -1) -> bool:
    d = fit_detail(machine, pack, ctx, vram_slots)
    return d["required_experts_gb"] <= d["ram_budget_gb"]


STREAMING_REASON = "experts don't fit in RAM; streaming from SSD"


def speed_usable(pack: dict) -> bool:
    """A fresh measured decode at or above the floor. IQ2_KT is below it until its CPU path is fixed."""
    return (pack.get("decode_tok_s") is not None
            and not pack.get("speed_stale", True)
            and pack["decode_tok_s"] >= DEFAULT_SPEED_FLOOR_TOK_S)


def select_pack(machine: Machine, packs=PACKS, ctx: int = DEFAULT_CONTEXT,
                vram_slots: int = -1) -> tuple[dict | None, str]:
    """-> (pack, why). First the best-quality pack that fits RAM and clears the speed floor; if none, the
    best-quality floor-clearing pack whose experts fit RAM+VRAM, streamed from SSD. Otherwise (None, '')."""
    available = [p for p in packs
                 if p.get("available", False)
                 and p["total_gb"] + DISK_RESERVE_GB <= machine.free_disk_gb]
    fitting = [p for p in available if pack_fits(machine, p, ctx, vram_slots) and speed_usable(p)]
    if fitting:
        pool = [p for p in fitting if p.get("recommended", True)] or fitting
        return max(pool, key=lambda p: (pack_quality(p), -p["total_gb"], -p["experts_gb"])), ""
    # Nothing fitting clears the floor. Fall back to a pack the combined RAM+VRAM can cover, with the
    # SSD streaming the difference; this is what keeps 128 GB on the measured 4 tok/s D3.
    combined_gb = machine.ram_gb + sum(machine.vram_gb)
    fallback = [p for p in available if speed_usable(p) and p["experts_gb"] <= combined_gb]
    if fallback:
        pool = [p for p in fallback if p.get("recommended", True)] or fallback
        return max(pool, key=lambda p: (pack_quality(p), -p["total_gb"], -p["experts_gb"])), STREAMING_REASON
    return None, ""


def choose_pack(machine: Machine, packs=PACKS, ctx: int = DEFAULT_CONTEXT,
                vram_slots: int = -1) -> dict | None:
    return select_pack(machine, packs, ctx, vram_slots)[0]


def pack_warning(pack: dict | None) -> str:
    if pack is None or pack.get("recommended", True):
        return ""
    return (f"sparsa: WARNING {pack['key']} is not recommended: it loops in practice "
            f"({pack['top1']:.1f} % top-1). It was picked only because nothing else fits.")


def pick_pack(machine: Machine | None = None, ctx: int = DEFAULT_CONTEXT,
              vram_slots: int = -1) -> dict | None:
    return choose_pack(machine or machine_specs(), ctx=ctx, vram_slots=vram_slots)


MMPROJ = "mmproj-MiniMax-M3.1-Flash-F16.gguf"
HF_REPO = "LyJonathan/MiniMax-M3.1-Flash-GGUF"
CHECKPOINT = "MiniMaxAI/MiniMax-M3.1-Flash"


def say(text: str = "") -> None:
    print(text, flush=True)


def find_engine(explicit: str | None) -> Path:
    """The engine binary: --engine, else the build folder of a source checkout, else PATH."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise SystemExit(f"sparsa: the engine is not at {p}")
        return p.resolve()
    for candidate in (ROOT / "build" / ENGINE_NAME, ROOT / "bin" / ENGINE_NAME, ROOT / "build-minimax" / ENGINE_NAME):
        if candidate.is_file():
            return candidate
    found = shutil.which(ENGINE_NAME) or shutil.which("strata-minimax")
    if found:
        return Path(found)
    raise SystemExit(
        f"sparsa: no {ENGINE_NAME} found next to this checkout (build/), on PATH, or given with --engine.\n"
        "        Build it: tools/minimax/build_windows.ps1 (Windows) or tools/minimax/build_linux.sh (Linux),\n"
        "        or download the release binary from https://github.com/LyJonathan/Sparsa/releases")


def find_model_guf(path_arg: str) -> tuple[Path, list[Path]]:
    """(the first part, every part in order) for a folder of parts or any one part."""
    p = Path(path_arg).expanduser()
    if p.is_file():
        folder = p.parent
        first = p
    elif p.is_dir():
        folder = p
        parts = sorted(folder.glob("*.gguf"))
        firsts = [q for q in parts if GGUF_PART in q.name]
        if not firsts:
            if len(parts) == 1:
                return parts[0].resolve(), parts
            raise SystemExit(f"sparsa: {folder} holds {len(parts)} .gguf files and no -00001-of- part; "
                             "name the model file itself")
        first = firsts[0]
    else:
        raise SystemExit(f"sparsa: {p} is not there")
    # every part of THIS set, in order (a folder may hold one model only; the name filter keeps it honest)
    stem = first.name.split(GGUF_PART)[0]
    parts = sorted(q for q in folder.glob(f"{stem}-*.gguf"))
    return first.resolve(), [q.resolve() for q in parts]


def find_tokenizer(model: Path, explicit: str | None) -> Path:
    """A folder with tokenizer.json: --tokenizer, the model folder, a tokenizer/ folder beside it."""
    tried = []
    for candidate in ([Path(explicit).expanduser()] if explicit else []) + [
            model.parent / "tokenizer", model.parent, model.parent.parent / "tokenizer"]:
        if (candidate / "tokenizer.json").is_file():
            return candidate.resolve()
        tried.append(str(candidate))
    raise SystemExit(
        "sparsa: the MiniMax tokenizer (tokenizer.json) was not found. The GGUF parts carry the token\n"
        "        ids but not the merges, so the server needs the small tokenizer files from the checkpoint:\n\n"
        f"            hf download {CHECKPOINT} tokenizer.json tokenizer_config.json chat_template.jinja \\\n"
        f"                --local-dir {model.parent / 'tokenizer'}\n\n"
        "        (or `python sparsa.py download tokenizer --dir <folder>`), then run this again.\n"
        f"        Looked in: {', '.join(tried)}")


def gpu_devices(explicit: str | None) -> list[int]:
    """The cards to use: --gpu 0,1, else every NVIDIA card nvidia-smi reports."""
    if explicit is not None:
        try:
            return [int(x) for x in explicit.replace(",", " ").split()]
        except ValueError:
            raise SystemExit(f"sparsa: --gpu wants card numbers like 0 or 0,1, not {explicit!r}") from None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [int(line) for line in out.stdout.split() if line.strip().isdigit()]


def prefill_batch_value(text: str) -> int:
    """`--prefill-batch`: `auto` (the default) asks for 4096 tokens a chunk and lets the engine shrink it to the
    free VRAM, `0`/`off` keeps the token path, and a number is that many tokens a chunk."""
    t = str(text).strip().lower()
    if t in ("", "auto", "default"):
        return 4096
    if t in ("off", "no", "none"):
        return 0
    try:
        n = int(t)
    except ValueError:
        raise SystemExit(f"sparsa: --prefill-batch wants auto, 0 or a token count, not {text!r}") from None
    if n < 0:
        raise SystemExit(f"sparsa: --prefill-batch cannot be negative ({n})")
    return n


def find_mmproj(model: Path, explicit: str | None) -> Path | None:
    """The vision file: --mmproj, else the one mmproj-*.gguf beside the model.  None keeps the engine text-only."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise SystemExit(f"sparsa: --mmproj {p} is not a file")
        return p
    found = sorted(model.parent.glob("mmproj-*.gguf"))
    return found[0] if found else None


def build_config(args) -> dict:
    if not args.model or str(args.model).strip().lower() in ("auto", "pick"):
        machine = machine_specs(args.models_dir)
        pack = choose_pack(machine, ctx=args.context, vram_slots=args.vram_slots)
        if pack is None:
            raise SystemExit(
                f"sparsa: no pack with a measured decode speed fits this machine ({machine.ram_gb:.0f} GB RAM, "
                f"{machine.free_disk_gb:.0f} GB free disk) in the RAM tier with the {RAM_FLOOR_GB:.0f} GB floor.\n"
                "        `python sparsa.py pick` shows the published tiers and what they need.")
        models_dir = Path(args.models_dir or Path.cwd() / "models").expanduser()
        folder = models_dir / pack["folder"]
        if not folder.is_dir():
            raise SystemExit(
                f"sparsa: this machine picks {pack['key']} ({pack['about']}), but {folder} is not downloaded.\n"
                f"        python sparsa.py download {pack['key']} --dir {models_dir}")
        say(f"sparsa: picked {pack['key']} for {machine.ram_gb:.0f} GB RAM "
            f"({pack['experts_gb']:.1f} GB of experts, {machine.gpu_count} GPU(s)): {folder}")
        if reason:
            say("sparsa: " + reason)
        warning = pack_warning(pack)
        if warning:
            say(warning)
        args.model = str(folder)
    first, parts = find_model_guf(args.model)
    engine = find_engine(args.engine)
    tokenizer = find_tokenizer(first, args.tokenizer)
    gpus = [] if args.cpu else gpu_devices(args.gpu)
    if not gpus and not args.cpu:
        raise SystemExit("sparsa: no NVIDIA card found. The MiniMax engine needs CUDA; pass --cpu only to test "
                         "a small file on the processor.")
    if len(parts) < EXPECTED_PARTS and not args.allow_nonstandard:
        say(f"sparsa: {first.parent} has {len(parts)} part(s) of the {EXPECTED_PARTS} a full quant has. "
            "A test slice needs --allow-nonstandard.")

    cfg_args = ["--model", str(first), "--max-context", str(args.context),
                "--minimax-dense-layers-gpu", "-1" if not args.cpu else "0"]
    if gpus:
        cfg_args += ["--gpus", ",".join(str(i) for i in range(len(gpus)))]
        cfg_args += ["--minimax-kv", "vram"]
    else:
        # host attention: the KV stays in ordinary memory and the layers stay on the CPU (the G3 quality runs'
        # path).  "ram" is a different mode (device attention over a RAM KV) and is not the same thing.
        cfg_args += ["--minimax-kv", "host"]
    if not args.cpu and not args.no_tiers:
        # --vram-slots -1 (the default) fills free VRAM after dense, KV, workspace and the reserve; 0 turns
        # expert slots off and makes the RAM tier hold every routed expert.
        cfg_args += ["--minimax-tiers", "--minimax-vram-slots-per-layer", str(args.vram_slots),
                     "--vram-reserve-mib", str(args.vram_reserve_mib)]
        if args.ram_gib:
            cfg_args += ["--memory-limit-mib", str(int(args.ram_gib * 1024))]
    prefill_batch = prefill_batch_value(args.prefill_batch)
    if not args.cpu and prefill_batch:
        # Chat always walks the batched prompt path: a prompt is read once per chunk instead of once per token,
        # which is the difference between ~14 and ~3.5 tok/s on a 200-400 token message (the token path re-reads
        # every routed expert for every token).  `auto` asks for 4096 and the engine shrinks the chunk to what each
        # card's free VRAM can hold after the weights and the KV; it prints the size it chose
        # (`INFO minimax-prefill batch=...`).  --prefill-batch 0 keeps the token path, N is a fixed chunk.
        # The expert projection stays on the int8 dots (the engine's `mmvq`): the full-model G3 gate for the
        # tensor-core path failed (KL64 1.235 -> 1.300, top-1 60.8 % -> 58.2 % over the six short prompts), so
        # `--prefill-expert-gemm tc` is a choice, not the default.
        cfg_args += ["--minimax-prefill-batch", str(prefill_batch),
                     "--minimax-prefill-expert-gemm", args.prefill_expert_gemm]
    if args.threads:
        cfg_args += ["--cpu-threads", str(args.threads)]
    if args.allow_nonstandard:
        cfg_args += ["--minimax-allow-nonstandard"]
    mmproj = None if getattr(args, "no_vision", False) else find_mmproj(first, args.mmproj)
    if mmproj is not None:
        cfg_args += ["--mmproj", str(mmproj)]
    cfg_args += list(args.engine_arg or [])

    log = Path(args.log).expanduser() if args.log else Path(args.config).with_suffix(".log")
    cfg = {
        "exe": str(engine),
        "args": cfg_args,
        "cwd": str(ROOT),
        "tokenizer": str(tokenizer),
        "model_name": MODEL_NAME,
        "format": "minimax_m3",
        "arch": "minimax_m3",
        # Deliberately no `sampling` block: serve/server.py's MODEL_CARD_SAMPLING is the one source of
        # MiniMax's published defaults, and a request's own fields still win over it.
        "log": str(log),
        "port": args.port,
        "drafting": "off",
    }
    if gpus:
        # several cards are a layer split; the engine places the layers by each card's free VRAM
        cfg["gpu"] = gpus if len(gpus) > 1 else gpus[0]
        if len(gpus) > 1:
            cfg["layer_split"] = "auto"
    if mmproj is not None:
        # The engine owns the ViT: the server writes the picture and counts its tokens, the engine projects it.
        cfg["vision"] = {"engine": True, "mmproj": str(mmproj), "max_tokens": 576}
    return cfg


def cmd_plan(args) -> int:
    cfg = build_config(args)
    say(json.dumps(cfg, indent=2))
    return 0


def prepare_libs(engine: Path) -> None:
    """The release puts the CUDA runtime libraries beside the engine; Linux needs them on the library path.

    Windows looks next to the .exe by itself, so this only does anything for the Linux tarball (whose libs
    sit in the engine's folder).  Harmless when the folder has no libraries."""
    if os.name == "nt":
        return
    folder = str(engine.parent)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if folder not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = folder + ((os.pathsep + current) if current else "")


def write_private_json(path: Path, data: dict) -> None:
    """Write the run settings owner-only: the file can hold a LAN API key. `os.open` creates it with 0600
    (before any bytes are written), and the chmod tightens a file left by an older version. Windows does not
    have POSIX modes; the user profile is already private there, so a chmod failure is ignored."""
    path = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(data, indent=1) + "\n")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def cmd_serve(args) -> int:
    from strata_cli.launch import access as access_mod

    config_path = Path(args.config).expanduser()
    if args.show_key:
        try:
            print(json.loads(config_path.read_text(encoding="utf-8-sig")).get("api_key") or
                  f"sparsa: {config_path} has no API key")
        except (OSError, ValueError):
            print(f"sparsa: no run settings at {config_path} yet")
        return 0
    cfg = build_config(args)
    prepare_libs(Path(cfg["exe"]))
    host = args.host or access_mod.host_for(args.access)
    key = ""
    if args.access != "local":
        try:
            key = str(json.loads(config_path.read_text(encoding="utf-8-sig")).get("api_key") or "") \
                if config_path.exists() else ""
        except (OSError, ValueError):
            key = ""
        if args.no_api_key:
            say("sparsa: WARNING no API key: anyone who can reach this port on your network can use the model.")
        else:
            key = key or access_mod.new_key()
            cfg["api_key"] = key
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_private_json(config_path, cfg)
    say(f"sparsa: run settings written to {config_path}")
    say(f"sparsa: engine {cfg['exe']}")
    say(f"sparsa: tokenizer {cfg['tokenizer']}")
    if cfg.get("vision"):
        say(f"sparsa: vision {cfg['vision']['mmproj']} (CPU encoder, ~1 min a picture)")
    elif not args.no_vision:
        say("sparsa: no mmproj beside the model: text only (`python sparsa.py download mmproj` adds pictures)")
    if args.access != "local":
        for ip in access_mod.lan_ips()[:2]:
            say(f"sparsa: LAN address http://{ip}:{cfg['port']}/v1"
                + (f"   (API key: {key})" if key else "   (no API key)"))
        note = access_mod.firewall_note(args.access)
        if note:
            say("sparsa: " + note)
        if key:
            say(f"sparsa: the key is kept in {config_path.name}; `--show-key` prints it again")
    if args.plan:
        say(json.dumps(cfg, indent=2))
        return 0
    python = sys.executable or "python"
    command = [python, str(ROOT / "serve" / "server.py"), "--engine", "strata",
               "--config", str(config_path), "--host", host, "--port", str(cfg["port"])]
    say("sparsa: starting; the model needs a minute or two to load (EXPERIMENTAL)")
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


def cmd_inspect(args) -> int:
    first, parts = find_model_guf(args.model)
    engine = find_engine(args.engine)
    prepare_libs(engine)
    say(f"sparsa: {first} ({len(parts)} part(s))")
    command = [str(engine), "--model", str(first), "--minimax-inspect"]
    return subprocess.call(command, cwd=ROOT)


def cmd_pick(args) -> int:
    machine = machine_specs(args.models_dir)
    pack, reason = select_pack(machine, ctx=args.context, vram_slots=args.vram_slots)
    if args.json:
        say(json.dumps({"machine": {"ram_gb": machine.ram_gb, "vram_gb": list(machine.vram_gb),
                                    "free_disk_gb": machine.free_disk_gb,
                                    "context": args.context, "vram_slots": args.vram_slots},
                        "pack": pack, "reason": reason, "warning": pack_warning(pack),
                        "speed_floor_tok_s": DEFAULT_SPEED_FLOOR_TOK_S}, indent=2))
        return 0
    say(f"sparsa: {machine.ram_gb:.0f} GB RAM, {machine.gpu_count} GPU(s) "
        f"({', '.join(f'{x:.0f} GB' for x in machine.vram_gb) or 'none'}), "
        f"{machine.free_disk_gb:.0f} GB free disk, {args.context} context")
    for p in PACKS:
        d = fit_detail(machine, p, args.context, args.vram_slots)
        fits = p["available"] and d["required_experts_gb"] <= d["ram_budget_gb"]
        disk = p["total_gb"] + DISK_RESERVE_GB <= machine.free_disk_gb
        state = "published" if p["available"] else "NOT PUBLISHED"
        top1 = "n/a" if p.get("top1") is None else f"{p['top1']:.1f}%"
        speed = "n/a" if p.get("decode_tok_s") is None else (
            f"{p['decode_tok_s']:.2f} tok/s" + ("*" if p.get("speed_stale") else ""))
        rec = "" if p.get("recommended", True) else "  NOT RECOMMENDED"
        mark = "  <- pick" if pack is not None and p["key"] == pack["key"] else ""
        say(f"  {p['key']:<10} top-1 {top1:>5}  {speed:<11}  experts {p['experts_gb']:7.1f} GB  "
            f"need {d['required_experts_gb']:7.1f}/{d['ram_budget_gb']:7.1f} GB  "
            f"pack {p['total_gb']:7.1f} GB  "
            f"{'fit' if fits else 'no fit'}, {'disk ok' if disk else 'too big for disk'}  "
            f"{state}{rec}{mark}")
        if p.get("measured_on"):
            say(f"             measured on {p['measured_on']}")
        if p.get("decode_tok_s") is not None:
            say(f"             speed: engine {p.get('speed_engine')} ({p.get('speed_commit')}), "
                f"ctx {p.get('speed_ctx')}, shared={'yes' if p.get('speed_shared') else 'no'}, "
                f"{'STALE' if p.get('speed_stale') else 'fresh'}"
                + (f" - {p['speed_note']}" if p.get("speed_note") else ""))
        if p.get("slice_decode_tok_s") is not None:
            say(f"             15-layer slice: {p['slice_decode_tok_s']:.2f} tok/s, engine "
                f"{p.get('slice_engine')}, ctx {p.get('slice_ctx')}, "
                f"{'STALE' if p.get('slice_stale') else 'fresh'}")
    if pack is None:
        say(f"sparsa: no pack clears the {DEFAULT_SPEED_FLOOR_TOK_S:.1f} tok/s speed floor with a usable "
            "measurement here")
    else:
        if reason:
            say("sparsa: " + reason + f" ({pack['key']})")
        warning = pack_warning(pack)
        if warning:
            say(warning)
    return 0


def cmd_download(args) -> int:
    if args.what is None:
        machine = machine_specs(args.dir)
        pack, reason = select_pack(machine, ctx=args.context, vram_slots=args.vram_slots)
        if pack is None:
            raise SystemExit(f"sparsa: no pack clears the {DEFAULT_SPEED_FLOOR_TOK_S:.1f} tok/s speed floor "
                             f"with a usable measurement here ({machine.ram_gb:.0f} GB RAM, "
                             f"{machine.free_disk_gb:.0f} GB free disk); `python sparsa.py pick` lists the tiers")
        say(f"sparsa: picked {pack['key']} for {machine.ram_gb:.0f} GB RAM: {pack['about']}")
        if reason:
            say("sparsa: " + reason)
        warning = pack_warning(pack)
        if warning:
            say(warning)
        args.what = pack["key"]
    if args.what in QUANTS and not next(p for p in PACKS if p["key"] == args.what)["available"]:
        raise SystemExit(f"sparsa: {args.what} is planned but not published yet; nothing to download")
    if args.what == "tokenizer":
        target = Path(args.dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        command = ["hf", "download", CHECKPOINT, "tokenizer.json", "tokenizer_config.json",
                   "chat_template.jinja", "--local-dir", str(target)]
    elif args.what == "mmproj":
        # One vision file for every tier, at the repository's root.  Point --dir at the model folder so the
        # file sits beside the parts: `serve` looks there (find_mmproj).
        target = Path(args.dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        command = ["hf", "download", HF_REPO, MMPROJ, "--local-dir", str(target)]
    else:
        folder, _what = QUANTS[args.what]
        target = Path(args.dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        command = ["hf", "download", HF_REPO, "--include", f"{folder}/*", "--local-dir", str(target)]
    say("sparsa: " + " ".join(command))
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if not hf or args.print_only:
        if not hf:
            say("sparsa: `hf` is not on PATH (python -m pip install -U huggingface_hub); the command above "
                "downloads the same files.")
        return 0
    return subprocess.call([hf, *command[1:]])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sparsa", description="Run MiniMax-M3.1-Flash on this PC (EXPERIMENTAL).")
    sp = ap.add_subparsers(dest="command", required=True)

    def serving(name: str, help_text: str):
        p = sp.add_parser(name, help=help_text)
        p.add_argument("--model", help="folder of GGUF parts, or the first part")
        p.add_argument("--tokenizer", help="folder holding tokenizer.json (default: beside the model)")
        p.add_argument("--engine", help=f"path to {ENGINE_NAME} (default: build/, PATH)")
        p.add_argument("--context", type=int, default=32768, help="context length (default 32768)")
        p.add_argument("--gpu", help="cards to use, as nvidia-smi numbers them (default: all)")
        p.add_argument("--cpu", action="store_true", help="run every matrix on the processor (tests only)")
        p.add_argument("--no-tiers", action="store_true", help="no RAM expert tier: every expert is read from the SSD")
        p.add_argument("--vram-slots", type=int, default=-1,
                       help="VRAM expert slots per layer (default -1: auto, free VRAM after dense/KV/workspace; "
                            "0 turns them off)")
        p.add_argument("--threads", type=int, help="CPU expert workers (default: the engine's own count)")
        p.add_argument("--vram-reserve-mib", type=int, default=700, help="VRAM kept free (default 700)")
        p.add_argument("--ram-gib", type=float, help="cap the RAM expert tier at this many GiB")
        p.add_argument("--prefill-batch", default="auto",
                       help="prompt chunk: auto (default: 4096, shrunk by the engine to the free VRAM), 0 = the "
                            "token path, or a token count")
        p.add_argument("--prefill-expert-gemm", choices=("auto", "mmvq", "tc"), default="mmvq",
                       help="how a batched chunk's routed experts are projected: mmvq (default: the int8 dots), "
                            "auto (tensor cores from 16 rows per expert, the int8 dots below), or tc. The full "
                            "IQ2_XS-D3 G3 gate failed for tc (KL64 1.235 -> 1.300, top-1 60.8%% -> 58.2%%), so it "
                            "stays opt-in")
        p.add_argument("--allow-nonstandard", action="store_true",
                       help="serve a test slice (fewer layers than a full model)")
        p.add_argument("--engine-arg", action="append", help="one more engine option, repeatable")
        p.add_argument("--mmproj", help="vision mmproj GGUF (default: the mmproj-*.gguf beside the model)")
        p.add_argument("--no-vision", action="store_true", help="ignore the mmproj and serve text only")
        p.add_argument("--port", type=int, default=8080)
        p.add_argument("--host", help="the address to listen on (default: 127.0.0.1, or 0.0.0.0 for --access lan)")
        p.add_argument("--access", choices=("local", "lan"), default="local",
                       help="local: this computer only (default); lan: every address, with an API key made for you")
        p.add_argument("--no-api-key", action="store_true",
                       help="serve on the LAN without a key (anyone who can reach the port can use the model)")
        p.add_argument("--show-key", action="store_true", help="print the saved LAN API key and exit")
        p.add_argument("--plan", action="store_true", help="write/print the run settings and start nothing")
        p.add_argument("--models-dir", help="where auto-picked packs live (default: ./models)")
        p.add_argument("--config", default=str(ROOT / "sparsa-minimax.json"), help="where the run settings go")
        p.add_argument("--log", help="engine log (default: the config's name with .log)")
        return p

    serving("serve", "write the run settings and start the server")
    serving("plan", "print the run settings and start nothing")

    ins = sp.add_parser("inspect", help="list what the engine reads from the GGUF (no CUDA, no loading)")
    ins.add_argument("--model", required=True)
    ins.add_argument("--engine")

    dl = sp.add_parser("download", help="download one quant folder, the tokenizer files, or the mmproj")
    dl.add_argument("what", nargs="?", choices=[*QUANTS, "tokenizer", "mmproj"],
                    help="what to download (omit it to pick the largest pack that fits this machine)")
    dl.add_argument("--dir", default="models", help="where the files go (default: ./models)")
    dl.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="context length for the automatic pick")
    dl.add_argument("--vram-slots", type=int, default=-1, help="VRAM expert slots per layer (-1 auto, 0 off)")
    dl.add_argument("--print-only", action="store_true", help="only print the command")

    pick = sp.add_parser("pick", help="show this machine's specs and the pack it would pick")
    pick.add_argument("--models-dir", help="where downloaded packs live (default: ./models)")
    pick.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="context length for the RAM-tier fit")
    pick.add_argument("--vram-slots", type=int, default=-1, help="VRAM expert slots per layer (-1 auto, 0 off)")
    pick.add_argument("--json", action="store_true", help="print JSON")

    args = ap.parse_args(argv)
    if not hasattr(args, "models_dir"):
        args.models_dir = None
    return {"serve": cmd_serve, "plan": cmd_plan, "inspect": cmd_inspect, "download": cmd_download,
            "pick": cmd_pick}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
