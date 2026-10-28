# Sparsa

**EXPERIMENTAL.** Development has slowed down but the project is still active.

Sparsa is a fork of [Strata](https://github.com/Niko1221/Strata) that runs **MiniMax-M3.1-Flash** on a
normal PC: one or two NVIDIA cards, system RAM, and the SSD for the experts that do not fit. The model
is a 427-billion-parameter mixture of experts, 60 layers, 57 of them with 128 experts each, four active
per token.

`src/minimax/` is a separate engine (`strata-minimax`) with MiniMax's sparse attention, router and
shared expert, paged expert tiers, and the vision encoder. The server is Strata's: OpenAI- and
Anthropic-compatible APIs, streaming, tool calls, a web page.

Most of the published quants do not load in stock llama.cpp, LM Studio or Ollama. They are on Hugging
Face in [LyJonathan/MiniMax-M3.1-Flash-GGUF](https://huggingface.co/LyJonathan/MiniMax-M3.1-Flash-GGUF),
13 parts per tier; point Sparsa at the folder. Top-1 is measured against an FP32 reference of the
original weights, 6 short prompts, 679 positions.

| Tier | Size | Top-1 | Use |
| --- | ---: | ---: | --- |
| `IQ1_S` | 94.8 GB | 44.3 % | Do not use: its output is incoherent (see limitations) |
| `IQ1_M` | 106.6 GB | 47.1 % | Smallest usable tier; 1-bit quality is degraded |
| `IQ2_KT` | 123.8 GB | 56.3 % | Best measured 2-bit per byte; Sparsa only |
| `IQ2_XXS` | 124.5 GB | 52.0 % | 2-bit that also loads in patched llama.cpp |
| `IQ2_XS` | 133.1 GB | quality not measured | 2-bit for patched llama.cpp |
| `IQ2_XS-D3` | 146.1 GB | 59.4 % | Default pick: best measured 2-bit, runs on a 128 GB PC |
| `IQ2_KL` | 152.7 GB | 59.8 % | Slightly better than D3; Sparsa only |
| `IQ3_K` | 191.2 GB | 69.8 % | 3-bit step up; Sparsa only |
| `IQ4_KS` | 233.6 GB | 74.4 % | Closest IQ tier to NVFP4; Sparsa only |
| `NVFP4` | 246.1 GB | 77.3 % | Best measured quality, a lossless copy of the checkpoint's 4-bit experts; Sparsa only |

`IQ2_XS` has only a patched-llama.cpp measurement on a harder eight-prompt set (46.8 %), not comparable
with the rows above. Pick the largest tier that fits in RAM + VRAM with ~10 GB to spare; Sparsa streams
the rest from the SSD.

## Run a release

You need an NVIDIA card (CUDA 12.8 or newer for an RTX 50 series; 12 GB is enough), the RAM the tier
does not fit in, and Python 3.10+. The latest build is
[v0.1.3](https://github.com/LyJonathan/Sparsa/releases/latest) (Windows and Linux, CUDA).

```sh
git clone https://github.com/LyJonathan/Sparsa.git
cd Sparsa
# unzip the release into bin/: bin/strata-minimax.exe (+ DLLs) or bin/strata-minimax
python -m pip install -r requirements.txt
python sparsa.py download iq2-xs-d3 --dir D:/models
python sparsa.py download tokenizer --dir D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3/tokenizer
python sparsa.py download mmproj --dir D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3   # optional, pictures
python sparsa.py serve --model D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3
```

`serve` loads the model (a minute or two), then serves the web page and APIs on
`http://127.0.0.1:8080`: OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, model
`minimax-m3.1-flash`. `--access lan` shares it with an API key; `--context 8192`, `--gpu 0,1`,
`--threads N`, `--plan` and `--engine PATH` are the useful flags. `python sparsa.py pick` prints the
tier this machine would choose; `python sparsa.py inspect --model FILE` lists what the engine reads
from a GGUF.

## Build from source

Windows (Visual Studio 2022+, CUDA 13.3, Ninja):

```powershell
$env:STRATA_GGML_DIR = 'C:\src\llama.cpp'    # a llama.cpp checkout at third_party/ggml/VERSION.txt
python -m pip install -r requirements-build.txt
powershell -File tools/minimax/build_windows.ps1 -Targets strata-minimax -CudaArch "75;80;86;89;120"
```

Linux (CUDA 12.8+, CMake, Ninja, C++17 compiler):

```sh
JOBS=8 STRATA_CUDA_ARCHS="75;80;86;89;120" tools/minimax/build_linux.sh
```

`120` is the RTX 50 series, `86` the RTX 30 series; each extra architecture costs build time. Without
`STRATA_GGML_DIR` CMake fetches the pinned llama.cpp commit. Design notes and measurements:
[tools/minimax/README.md](tools/minimax/README.md).

## Measured speed

v0.1.2 on 2x RTX 5070 12 GB, Ryzen 9 5900X, 128 GB DDR4, one NVMe, full `IQ2_XS-D3` at `serve`
defaults: 4.2 tok/s decoding (240 ms/token); prefill 13.5 tok/s on a 411-token prompt, 41.5 tok/s on
4,686 tokens, 44.3 tok/s on 32K. The same prompt took 33.8 s in one run and 59.9 s in another: the
tier is drive-bound, so read these as "half a minute for a chat message".

## Known limitations

- **Speed.** A few tokens a second on the measured box, and it depends on the SSD.
- **Rare stall.** About 5 % of runs stall in the CPU-expert prompt path; the cause is not known. Because
  of it, [PR #34](https://github.com/LyJonathan/Sparsa/pull/34) (+47 % prompt speed when prompt experts
  run on CPU, bit-identical) is not merged.
- **IQ1_S** produces incoherent output in practice.
- **Context.** 8K-32K works; above 32K expect it to be slow or to fail.
- **Vision.** Images only, no video; the encoder is CPU fp32 and takes 64-80 s per 640x480 image.
- **Tool calls** use MiniMax's XML form and are parsed whole, so arguments are not streamed.
- **CI** is a Linux build only (no GPU, no model).

## Contributing

PRs that improve Sparsa are welcome, for example the open CPU-expert prompt stall, speed, or a
measured tier's quality: see [issues](https://github.com/LyJonathan/Sparsa/issues) and
[pull requests](https://github.com/LyJonathan/Sparsa/pulls).

## Upstream Strata

The Qwen side of this repository is upstream's and still works: the `strata` Terminal, the desktop
app, `docs/DETAILS.md`. Upstream's README is kept as [README-STRATA.md](README-STRATA.md), with
translated copies beside it.

## Licence and credits

Sparsa is MIT (`LICENSE`); [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the code carried from
other projects. The model files are under the
[MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/LICENSE) and
are not part of this repository.

- [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) - the model, its chat template and its
  vision encoder.
- [Niko1221/Strata](https://github.com/Niko1221/Strata) and contributors - the fork's base: the server,
  the Terminal, the app, the Qwen engine.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) - the GGUF format, quantization and the CUDA
  kernels the engine reuses; stb_image comes from it too.
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by Iwan Kawrakow - the IQ2_KT, IQ2_KL,
  IQ3_K and IQ4_KS formats the expert tiers port.
- [Project Maya](https://github.com/mw00/project-maya) (Peasant Smith and contributors, MIT) - the GPTQ
  error-feedback rounding the IQ quantizer ports, and the tier ideas the engine measures against.
- [agentionai/llama.cpp](https://github.com/agentionai/llama.cpp) - the trellis expert types.
- [Strata-DS](https://github.com/Jackwwg83/Strata-DS) (Jackwwg83) - the server front end the MiniMax
  tokenizer handling follows.
- [Xolotl](https://github.com/Indras-Mirror/Xolotl) (MIT) - the resident-expert routing bias the opt-in
  route nudge follows.
- Unsloth - the dynamic quant-mix idea the low tiers borrow; ubergarm - ik_llama.cpp quant recipes.
