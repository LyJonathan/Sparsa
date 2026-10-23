# MiniMax engine tools (P2a scaffold)

The engine lives in `src/minimax/` and `include/strata/minimax/`. This phase loads and validates a GGUF's directory;
there is no forward pass yet.

## Build (Windows, VS 2026 + CUDA 13.3, sm_120)

```powershell
pwsh -File tools/minimax/build_windows.ps1
```

The script maps the worktree to `M:` because nvcc drops include paths that contain the apostrophe in
`Jon's Projects`. It sets `CUDA_PATH` to 13.3 explicitly (the process environment may point at 13.1), configures
`build/` with `-DCMAKE_CUDA_ARCHITECTURES=120`, and builds `strata` and `strata-minimax`. Set `STRATA_GGML_DIR` to a
llama.cpp checkout at the pinned `third_party/ggml/VERSION.txt` commit; without it CMake fetches that commit. The
MiniMax-patched `_baseline/llama.cpp` is not a Strata build dependency (its MMQ API differs from the pinned one).

## Inspect a GGUF

```powershell
build\strata-minimax.exe --model model.gguf --minimax-inspect
```

Prints every expected `minimax-m3` tensor (name, GGML type, shape, bytes, offset, class) and the byte totals for
dense / experts / embd / output. The inspect path uses the structural check only, so it can describe a file that is
not MiniMax-M3.1-Flash; `--serve` adds the exact M3.1 fingerprint check and refuses anything else.

## Serve skeleton

```powershell
build\strata-minimax.exe --model model.gguf --serve --max-context 32768
```

Prints `INFO engine=minimax-m3 ...` and `READY <ctx> stop`, refuses `GEN`/`GENI` with
`ERR minimax engine: forward pass not implemented yet` (no `DONE`), and handles `STOP` and `QUIT`. Unknown options
are ignored in `--serve` mode, because the server's config is shared with the Qwen families.

`--minimax-allow-nonstandard` serves a reduced test GGUF (the synthetic file below); `--serve` otherwise refuses it.

## Synthetic GGUF (loader test)

```powershell
python tools/minimax/make_synthetic_gguf.py synthetic.gguf
python -m pytest tools/minimax/test_inspect.py -q
```

The generator uses gguf-py from `_baseline/llama.cpp/gguf-py` (or `--gguf-py`) and writes two layers with reduced
dimensions and every real `minimax-m3` tensor name/shape relationship. The tests check the inspect table, a missing
tensor, a wrong shape, the M3.1 refusal, and the serve handshake. They skip when `strata-minimax.exe` is not built.
