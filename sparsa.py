"""sparsa.py - run MiniMax-M3.1-Flash on this PC.

Sparsa is a fork of Strata that adds a MiniMax-M3.1-Flash engine, ``strata-minimax``.  Strata's own
``strata`` command knows the Qwen families; this small program writes the run config a MiniMax model
needs and hands it to the same Python server (``serve/server.py``: OpenAI ``/v1/chat/completions``,
Anthropic ``/v1/messages``, streaming, tool calls, the web page).

    python sparsa.py serve --model D:\\models\\MiniMax-M3.1-Flash-IQ2_XS-D3
    python sparsa.py serve --model model.gguf --context 8192 --gpu 0
    python sparsa.py plan  --model model.gguf          # print the settings, start nothing
    python sparsa.py download iq2-xs-d3 --dir D:\\models
    python sparsa.py inspect --model model.gguf        # what the engine reads from the file

The model argument is the folder the GGUF parts were downloaded into, or the first part.  The
tokenizer is needed too (``tokenizer.json`` from the checkpoint, ~10 MB); ``download tokenizer``
fetches it, and ``serve`` says where it looked when it cannot find it.

EXPERIMENTAL.  Decode is slow (a few tokens a second on two consumer cards), long context is work
in progress, and vision is not implemented.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE_NAME = "strata-minimax.exe" if os.name == "nt" else "strata-minimax"
MODEL_NAME = "minimax-m3.1-flash"
EXPECTED_PARTS = 13
GGUF_PART = "-00001-of-"

# LyJonathan/MiniMax-M3.1-Flash-GGUF: folder name -> what it is.  Kept here so `download` can name
# the exact `hf` command without a network call.
QUANTS = {
    "iq2-xs": ("MiniMax-M3.1-Flash-IQ2_XS", "IQ2_XS experts; 133.1 GB; loads in llama.cpp (patched) and Sparsa"),
    "iq2-xs-d3": ("MiniMax-M3.1-Flash-IQ2_XS-D3", "IQ2_XS gate/up + IQ3_XXS down; 146.1 GB; measured quality"),
    "iq3-k": ("MiniMax-M3.1-Flash-IQ3_K", "IQ3_K experts; 191.2 GB; Sparsa only"),
    "iq4-ks": ("MiniMax-M3.1-Flash-IQ4_KS", "IQ4_KS experts; 233.6 GB; Sparsa only"),
    "iq2-kl": ("MiniMax-M3.1-Flash-IQ2_KL", "IQ2_KL experts; 152.7 GB; Sparsa only"),
    "nvfp4": ("MiniMax-M3.1-Flash-NVFP4", "the checkpoint's NVFP4 expert values, lossless; 246.1 GB; Sparsa only"),
}
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
    if not args.model:
        raise SystemExit("sparsa: --model is needed (the folder of GGUF parts, or the first part)")
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
        # The VRAM expert slots are OFF by default (0): the auto count is measured to leave too little VRAM for
        # the batched-prefill workspace, and with 128 experts top-4 most routes are not resident anyway; the RAM
        # tier is what makes the token fast.  --vram-slots N turns them on.
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
        "sampling": {"temperature": 1.0, "top_p": 0.95},
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
    config_path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    say(f"sparsa: run settings written to {config_path}")
    say(f"sparsa: engine {cfg['exe']}")
    say(f"sparsa: tokenizer {cfg['tokenizer']}")
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


def cmd_download(args) -> int:
    if args.what == "tokenizer":
        target = Path(args.dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        command = ["hf", "download", CHECKPOINT, "tokenizer.json", "tokenizer_config.json",
                   "chat_template.jinja", "--local-dir", str(target)]
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
        p.add_argument("--vram-slots", type=int, default=0,
                       help="VRAM expert slots per layer (default 0: off; the auto count starves prefill)")
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
        p.add_argument("--config", default=str(ROOT / "sparsa-minimax.json"), help="where the run settings go")
        p.add_argument("--log", help="engine log (default: the config's name with .log)")
        return p

    serving("serve", "write the run settings and start the server")
    serving("plan", "print the run settings and start nothing")

    ins = sp.add_parser("inspect", help="list what the engine reads from the GGUF (no CUDA, no loading)")
    ins.add_argument("--model", required=True)
    ins.add_argument("--engine")

    dl = sp.add_parser("download", help="download one quant folder, or the tokenizer files")
    dl.add_argument("what", choices=[*QUANTS, "tokenizer"])
    dl.add_argument("--dir", required=True, help="where the files go")
    dl.add_argument("--print-only", action="store_true", help="only print the command")

    args = ap.parse_args(argv)
    return {"serve": cmd_serve, "plan": cmd_plan, "inspect": cmd_inspect, "download": cmd_download}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
