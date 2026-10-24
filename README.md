# Sparsa

![EXPERIMENTAL](https://img.shields.io/badge/STATUS-EXPERIMENTAL-red?style=for-the-badge)
![MiniMax M3.1 Flash](https://img.shields.io/badge/MODEL-MINIMAX--M3.1--FLASH-red?style=for-the-badge)

> [!CAUTION]
> **Experimental.** This is a working preview, not a finished product. It loads and answers, it is
> slow, and some parts are not done. Expect bugs, and expect the settings to change.
>
> **Most of the quants do not load in stock llama.cpp, LM Studio or Ollama.** Use this engine.

Sparsa is a fork of [Strata](https://github.com/Niko1221/Strata) that adds a runtime for
**MiniMax-M3.1-Flash**: a 427-billion-parameter mixture-of-experts model with 60 layers, 57 of them
holding 128 experts each, four of which run per token. It runs on a normal PC: one or two NVIDIA
cards, the system RAM, and the SSD for whatever does not fit.

It is a real fork, not a wrapper: `src/minimax/` is a separate engine (`strata-minimax`) with
MiniMax's sparse attention, its router and shared expert, the expert tiers, the vision encoder, and a
serve protocol the Python server speaks. The server is Strata's: OpenAI- and Anthropic-compatible
APIs, streaming, tool calls, a web page.

## Quants

All ten tiers are on Hugging Face in
[LyJonathan/MiniMax-M3.1-Flash-GGUF](https://huggingface.co/LyJonathan/MiniMax-M3.1-Flash-GGUF),
13 parts each; point Sparsa at the folder. Sizes are the published bytes, bpw is over the 426.2 B
language-model parameters, and top-1 is the quant's top next-token agreeing with the full-precision
model (six short prompts, 679 positions; higher is better).

| quant | size | bpw | loads in | top-1 |
| --- | ---: | ---: | --- | ---: |
| `IQ1_S` | 94.8 GB | 1.78 | Sparsa, llama.cpp (patched) | pending |
| `IQ1_M` | 106.6 GB | 2.00 | Sparsa, llama.cpp (patched) | 47.1 % |
| `IQ2_KT` | 123.8 GB | 2.32 | **Sparsa only** | pending |
| `IQ2_XXS` | 124.5 GB | 2.34 | Sparsa, llama.cpp (patched) | pending |
| `IQ2_XS` | 133.1 GB | 2.50 | Sparsa, llama.cpp (patched) | 46.8 % * |
| `IQ2_XS-D3` | 146.1 GB | 2.74 | Sparsa, llama.cpp (patched) | 59.4 % |
| `IQ2_KL` | 152.7 GB | 2.87 | **Sparsa only** | 59.8 % |
| `IQ3_K` | 191.2 GB | 3.59 | **Sparsa only** | 69.8 % |
| `IQ4_KS` | 233.6 GB | 4.38 | **Sparsa only** | 74.4 % |
| `NVFP4` | 246.1 GB | 4.62 | **Sparsa only** | 77.3 % |

\* `IQ2_XS` has no Sparsa measurement: 46.8 % is patched llama.cpp on a harder eight-prompt set, so it
is not comparable with the rows above (`IQ2_XS-D3` on that set: 53.8 %).

`Sparsa only` means the expert types are Strata's own ids (ik_llama.cpp's IQ2_KT / IQ2_KL / IQ3_K /
IQ4_KS, and the checkpoint's NVFP4 values); upstream llama.cpp cannot read those files. `llama.cpp
(patched)` means the small MiniMax-M3 patch that ships with Sparsa. The 1-2-bit tiers are noticeably
degraded: the experts were trained as 4-bit (NVFP4).

Pick the biggest that fits in RAM + VRAM with ~10 GB to spare; whatever does not fit is streamed from
the SSD, which works but is slower.

## Vision

`mmproj-MiniMax-M3.1-Flash-F16.gguf` (1.73 GB) adds image input for every quant: OpenAI `image_url`
parts and Anthropic image blocks, as a URL, a base64 data URL or a local path. The ViT runs on the CPU
in fp32 and takes 64-80 s per 640x480 image; a full-model check answered "The object is a red circle."
for a red circle. Images only, no video.

## Quickstart

You need an NVIDIA card (CUDA 12.8 or newer for an RTX 50 series; 12 GB is enough), the RAM the tier
does not fit in, and Python 3.10+.

**1. Get the runtime.** Either unzip a release into a checkout of this repository so the engine sits
in `bin/`, or build it (below). The release has `strata-minimax.exe` (Windows) or `strata-minimax`
(Linux) plus the CUDA libraries it needs.

```sh
git clone https://github.com/LyJonathan/Sparsa.git
cd Sparsa
# put the release's files in bin/: bin/strata-minimax.exe (+ DLLs) or bin/strata-minimax
python -m pip install -r requirements.txt
```

**2. Download a model, the tokenizer and the vision file.** The GGUF carries the token ids but not the
BPE merges, so the small tokenizer files come from the checkpoint (about 10 MB). `download` uses `hf`
(`python -m pip install -U huggingface_hub`) and prints the same command when it is absent.

```sh
python sparsa.py download iq2-xs-d3 --dir D:/models
M=D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3
python sparsa.py download tokenizer --dir $M/tokenizer
python sparsa.py download mmproj --dir $M          # picture input; skip it for text only
```

**3. Serve it.**

```sh
python sparsa.py serve --model $M
```

It prints the run settings it wrote and the files it found, starts the engine (a minute or two to load),
then:

- web page and API: `http://127.0.0.1:8080/`
- OpenAI: `http://127.0.0.1:8080/v1/chat/completions`, model `minimax-m3.1-flash`
- Anthropic: `http://127.0.0.1:8080/v1/messages`

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "minimax-m3.1-flash",
  "messages": [{"role": "user", "content": "Write one sentence about the sea."}],
  "max_tokens": 200, "stream": true
}'
```

A picture is an ordinary content part:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is in this image?"},
  {"type": "image_url", "image_url": {"url": "C:/pics/blue-square.png"}}]}
```

Useful flags: `--context 8192` (default 32768; long context is work in progress), `--gpu 0` or
`--gpu 0,1`, `--ram-gib N` to cap the RAM expert tier (the engine's `--memory-limit-mib`, reachable as
`--engine-arg --memory-limit-mib N`),
`--threads N`, `--mmproj PATH`, `--no-vision`, `--port`, `--plan` to print the settings without
starting, `--engine PATH`. `python sparsa.py inspect --model FILE` lists what the engine reads from a
GGUF.

### Share it on your network

By default the server listens on `127.0.0.1` only. `--access lan` makes an API key and binds every
address:

```sh
python sparsa.py serve --model $M --access lan
# sparsa: LAN address http://192.168.1.50:8080/v1   (API key: sk-strata-...)
```

Send the key as `Authorization: Bearer ...`, or as `x-api-key` from Anthropic clients. The key is kept
in `sparsa-minimax.json` (which `.gitignore` covers); `--show-key` prints it again, and `--no-api-key`
serves without one, with a warning: anyone who can reach the port could then use the model. Keep it on
your own network; there is no user account, just the one key.

The server is the Qwen families' server too. `strata` (the Terminal) still manages those;
`python sparsa.py` is the MiniMax path.

## Speed

Measured in v0.1.2 on 2x RTX 5070 12 GB, Ryzen 9 5900X, 128 GB DDR4, one NVMe: full
`IQ2_XS-D3` (146.1 GB), `sparsa.py serve` defaults (one 4096-token prompt chunk, the int8 expert dots,
the RAM tier capped at 84 GiB), the box idle.

| what | result |
| --- | ---: |
| decode, 64 greedy tokens | **4.2 tok/s** (240 ms/token) |
| prefill, 411-token prompt | 30.5 s = **13.5 tok/s** |
| prefill, 4,686 tokens | 112.9 s = **41.5 tok/s** |
| prefill, 32 K tokens | 739.6 s = **44.3 tok/s** |

Every number is drive-bound: the experts not in the RAM tier (7.8 per token) come off the SSD, so the
same prompt took 33.8 s in one run and 59.9 s in another. Read the prefill rows as "about half a minute
for a chat message, a minute or two for a few thousand tokens". Batching is what makes chat usable: the
same 411-token message through the per-token path (`--prefill-batch 0`) takes 124 s.

## Build from source

Windows (Visual Studio 2022+, CUDA 13.3, Ninja):

```powershell
$env:STRATA_GGML_DIR = 'C:\src\llama.cpp'    # a llama.cpp checkout at third_party/ggml/VERSION.txt
python -m pip install -r requirements-build.txt
powershell -File tools/minimax/build_windows.ps1 -Targets strata-minimax -CudaArch "75;80;86;89;120"
# the worktree must not sit under a path with an apostrophe: nvcc drops those include paths, and
# the script maps the tree to a drive letter (-SubstDrive Q: -GgmlSubstDrive R:)
```

Linux (CUDA 12.8+, CMake, Ninja, a C++17 compiler):

```sh
JOBS=8 STRATA_CUDA_ARCHS="75;80;86;89;120" tools/minimax/build_linux.sh
```

`-CudaArch`/`STRATA_CUDA_ARCHS` are the card architectures: `120` for the RTX 50 series, `86` for an
RTX 30 series, and so on; a range costs build time per architecture. Without `STRATA_GGML_DIR` CMake
fetches the pinned llama.cpp commit (network needed). Every engine flag and the measurements behind
them: [tools/minimax/README.md](tools/minimax/README.md).

## Known limits

- **Speed.** A few tokens a second on the measured box, and it depends on the SSD.
- **Long context.** The model supports 1M tokens; this engine has a working 8K-32K path. Above 32K
  expect it to be slow or to fail.
- **Vision.** Images only, no video. The encoder is CPU fp32 and takes 64-80 s a picture at 640x480.
- **Tool calls** work through the XML form MiniMax's template uses, and are parsed whole (arguments
  are not streamed as fragments).
- **Sampling** is the card's (temperature 1.0, top_p 0.95) unless the request says otherwise.
- **CI.** GitHub Actions do not run in this fork (the account's Actions budget); the checks in this
  tree were run locally.

## The original Strata

The Qwen side of this repository is upstream's and still works: the `strata` Terminal, the desktop
app, the Qwen3.8-Flash-Next models, `docs/DETAILS.md`, `AGENTS.md`. Upstream's own README is kept as
[README-STRATA.md](README-STRATA.md) (and the translated README.*.md files beside it); start there for
the Qwen models, or at [Niko1221/Strata](https://github.com/Niko1221/Strata).

## Licence and credits

Sparsa is MIT (`LICENSE`). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists what it carries from
other projects and their licences. The model files are under the
[MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/LICENSE):
commercial use requires displaying "Built with MiniMax M3.1-Flash" and notifying api@minimax.io, and
its prohibited uses apply. They are not part of this repository.

- [Niko1221/Strata](https://github.com/Niko1221/Strata) and its contributors - the fork's base: the
  server, the Terminal, the app, the Qwen engine.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) - the GGUF format, quantization, and the
  CUDA kernels the engine reuses.
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by Iwan Kawrakow - the IQ2_KT / IQ3_K /
  IQ4_KS / IQ2_KL formats the expert tiers port.
- [agentionai/llama.cpp](https://github.com/agentionai/llama.cpp) ("Gyro") - the trellis expert types.
- [Strata-DS](https://github.com/Jackwwg83/Strata-DS) (Jackwwg83) - the DeepSeek server front end the
  MiniMax tokenizer and parser are modelled on, and the RAM-tier measurements this engine compares to.
- [Project Maya](https://github.com/mw00/project-maya) (Peasant Smith and contributors, MIT) - the
  DeepSeek engine whose pool spin, RAM-tier headroom and in-place reads were measured against here.
- [Xolotl](https://github.com/Indras-Mirror/Xolotl) (MIT) - the resident-expert routing bias this
  engine's opt-in route nudge follows.
- Unsloth - the dynamic quant-mix idea the low tiers borrow.
- MiniMax - the model, its chat template and its vision encoder.
