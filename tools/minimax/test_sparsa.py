"""Tests for sparsa.py's run settings (no engine, no GPU, no model download).

The GGUF parts and the tokenizer are empty files: only their names and presence matter here.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sparsa  # noqa: E402


def make_files(tmp_path, parts=13, tokenizer=True):
    model = tmp_path / "MiniMax-M3.1-Flash-IQ2_XS-D3"
    model.mkdir()
    for i in range(1, parts + 1):
        (model / f"MiniMax-M3.1-Flash-IQ2_XS-D3-{i:05d}-of-{parts:05d}.gguf").write_bytes(b"GGUF")
    if tokenizer:
        tok = model / "tokenizer"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}", encoding="utf-8")
    engine = tmp_path / "strata-minimax.exe"
    engine.write_bytes(b"")
    return model, engine


def plan(model, engine, tmp_path, *extra):
    """Run `sparsa.py plan` and return (the config it printed, its other stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = sparsa.main(["plan", "--model", str(model), "--engine", str(engine),
                            "--config", str(tmp_path / "run.json"), *extra])
    assert code == 0
    text = out.getvalue()
    return json.loads(text[text.index("{"):]), text


def test_plan_is_the_minimax_format_and_starts_at_part_one(tmp_path):
    model, engine = make_files(tmp_path)
    cfg, _ = plan(model, engine, tmp_path)
    assert cfg["format"] == "minimax_m3"
    assert cfg["model_name"] == "minimax-m3.1-flash"
    assert cfg["drafting"] == "off"
    assert cfg["args"][cfg["args"].index("--model") + 1].endswith("-00001-of-00013.gguf")
    # the measured arm: the RAM tier on, the VRAM expert slots off (auto starves prefill)
    assert "--minimax-tiers" in cfg["args"]
    assert cfg["args"][cfg["args"].index("--minimax-vram-slots-per-layer") + 1] == "0"
    # chat reads the prompt in one batched chunk by default ("auto" -> 4096, the engine shrinks it to what fits)
    assert cfg["args"][cfg["args"].index("--minimax-prefill-batch") + 1] == "4096"
    # ... with the int8 expert dots: the tensor-core arm failed the full-model G3 gate, so it is opt-in
    assert cfg["args"][cfg["args"].index("--minimax-prefill-expert-gemm") + 1] == "mmvq"


def test_prefill_batch_takes_auto_a_count_or_off(tmp_path):
    model, engine = make_files(tmp_path)
    cfg, _ = plan(model, engine, tmp_path, "--prefill-batch", "512")
    assert cfg["args"][cfg["args"].index("--minimax-prefill-batch") + 1] == "512"
    for off in ("0", "off"):
        cfg, _ = plan(model, engine, tmp_path, "--prefill-batch", off)
        assert "--minimax-prefill-batch" not in cfg["args"]
        assert "--minimax-prefill-expert-gemm" not in cfg["args"]
    cfg, _ = plan(model, engine, tmp_path, "--prefill-expert-gemm", "mmvq")
    assert cfg["args"][cfg["args"].index("--minimax-prefill-expert-gemm") + 1] == "mmvq"
    with pytest.raises(SystemExit) as e:
        plan(model, engine, tmp_path, "--prefill-batch", "lots")
    assert "--prefill-batch" in str(e.value)


def test_a_slice_is_named_and_needs_the_flag(tmp_path):
    model, engine = make_files(tmp_path, parts=2)
    _cfg, text = plan(model, engine, tmp_path)
    assert "test slice" in text
    cfg, _ = plan(model, engine, tmp_path, "--allow-nonstandard")
    assert "--minimax-allow-nonstandard" in cfg["args"]


def test_cpu_test_path_turns_the_gpu_flags_off(tmp_path):
    model, engine = make_files(tmp_path)
    cfg, _ = plan(model, engine, tmp_path, "--cpu")
    assert "--gpus" not in cfg["args"]
    assert cfg["args"][cfg["args"].index("--minimax-dense-layers-gpu") + 1] == "0"
    assert "--minimax-tiers" not in cfg["args"]


def test_missing_tokenizer_says_how_to_get_it(tmp_path):
    model, engine = make_files(tmp_path, tokenizer=False)
    with pytest.raises(SystemExit) as e:
        plan(model, engine, tmp_path)
    assert "hf download MiniMaxAI/MiniMax-M3.1-Flash" in str(e.value)


def test_download_prints_the_hf_command(tmp_path, capsys):
    assert sparsa.main(["download", "iq3-k", "--dir", str(tmp_path), "--print-only"]) == 0
    assert "MiniMax-M3.1-Flash-IQ3_K/*" in capsys.readouterr().out


def test_download_mmproj_names_the_one_vision_file(tmp_path, capsys):
    assert sparsa.main(["download", "mmproj", "--dir", str(tmp_path), "--print-only"]) == 0
    out = capsys.readouterr().out
    assert "mmproj-MiniMax-M3.1-Flash-F16.gguf" in out
    assert "LyJonathan/MiniMax-M3.1-Flash-GGUF" in out


def test_a_mmproj_beside_the_model_turns_vision_on(tmp_path, capsys):
    model, engine = make_files(tmp_path)
    mmproj = model / "mmproj-MiniMax-M3.1-Flash-F16.gguf"
    mmproj.write_bytes(b"GGUF")
    cfg, text = plan(model, engine, tmp_path)
    assert cfg["vision"]["engine"] is True
    assert cfg["vision"]["mmproj"] == str(mmproj)
    assert cfg["args"][cfg["args"].index("--mmproj") + 1] == str(mmproj)
    assert "vision" in text


def test_no_vision_keeps_the_mmproj_out(tmp_path):
    model, engine = make_files(tmp_path)
    (model / "mmproj-MiniMax-M3.1-Flash-F16.gguf").write_bytes(b"GGUF")
    cfg, _ = plan(model, engine, tmp_path, "--no-vision")
    assert "vision" not in cfg
    assert "--mmproj" not in cfg["args"]


def test_explicit_mmproj_path_is_used(tmp_path):
    model, engine = make_files(tmp_path)
    other = tmp_path / "elsewhere" / "vision.gguf"
    other.parent.mkdir()
    other.write_bytes(b"GGUF")
    cfg, _ = plan(model, engine, tmp_path, "--mmproj", str(other))
    assert cfg["vision"]["mmproj"] == str(other)


def test_lan_access_makes_a_key_and_reuses_it(tmp_path, capsys):
    model, engine = make_files(tmp_path)
    config = tmp_path / "run.json"
    argv = ["serve", "--model", str(model), "--engine", str(engine), "--config", str(config),
            "--access", "lan", "--plan"]
    assert sparsa.main(argv) == 0
    first = json.loads(config.read_text(encoding="utf-8"))
    assert first["api_key"].startswith("sk-strata-")
    out = capsys.readouterr().out
    assert "172.0.0.1" not in out
    assert sparsa.main(argv) == 0
    second = json.loads(config.read_text(encoding="utf-8"))
    assert second["api_key"] == first["api_key"]


def test_local_access_writes_no_key_and_binds_loopback(tmp_path, capsys):
    model, engine = make_files(tmp_path)
    config = tmp_path / "run.json"
    assert sparsa.main(["serve", "--model", str(model), "--engine", str(engine),
                        "--config", str(config), "--plan"]) == 0
    assert "api_key" not in json.loads(config.read_text(encoding="utf-8"))
    assert "LAN address" not in capsys.readouterr().out


def test_no_api_key_on_the_lan_says_what_that_means(tmp_path, capsys):
    model, engine = make_files(tmp_path)
    config = tmp_path / "run.json"
    assert sparsa.main(["serve", "--model", str(model), "--engine", str(engine), "--config", str(config),
                        "--access", "lan", "--no-api-key", "--plan"]) == 0
    assert "api_key" not in json.loads(config.read_text(encoding="utf-8"))
    assert "anyone" in capsys.readouterr().out
