# Sparsa

**Run MiniMax-M3.1-Flash, a 427-billion-parameter model, on a normal PC.** Two mid-range graphics cards, the
RAM you already have, and an SSD.

**EXPERIMENTAL.** Development has slowed down but the project is still active.

| Measured on two 16 GB cards (RTX 4060 Ti + RTX 5060 Ti), Ryzen 9 5900X, 121 GB DDR4, one NVMe | |
| --- | ---: |
| Prompt, 4,000 tokens | **245 tokens/s** (16 s) |
| Prompt, 16,000 tokens | **245 tokens/s** |
| Prompt, 500 tokens | **~6 s** |
| Writing (decode), at a 32K context | **~8 tokens/s** (8.7 at best) |
| Model | `IQ2_XXS`, 124.5 GB: about 4x the VRAM of both cards together |

At the start of this round of work, the same box read a 4,000-token prompt at 50 tokens/s. What changed is listed in
[PR #50](https://github.com/LyJonathan/Sparsa/pull/50). Every number there was measured, and every new kernel was
checked against the one it replaces.

## Why it is fast

MiniMax-M3.1-Flash has 60 layers and 128 experts in each MoE layer, with 4 active per token. Sparsa keeps the hot
experts on the GPUs, most of the rest in pinned RAM and the remainder on the SSD, and moves each one to where it is
cheapest to run:

- **Tensor cores everywhere in the prompt pass:** expert GEMMs, the sparse-attention kernel (WMMA) and the
  indexer's block scores (48x faster than before).
- **Both PCIe links, both cards:** a card on a slow link sends its layers' experts through the faster card, while
  computing the experts it already holds.
- **Copies overlap compute:** an asynchronous 8-slot ring, SSD reads on their own thread into pinned staging, and
  normed inputs sent as fp16.
- **Short prompts lean on the CPU:** experts with a few tokens run from RAM on the CPU (faster than a PCIe copy)
  while the cards stream the rest.
- **Decode follows the conversation:** per-card VRAM expert slots, a routing profile shipped with every tier, and a
  prompt cache that keeps the longest shared prefix.

## The model files

They are on Hugging Face in
[LyJonathan/MiniMax-M3.1-Flash-GGUF](https://huggingface.co/LyJonathan/MiniMax-M3.1-Flash-GGUF): one folder per tier.
Point Sparsa at the folder. Most tiers do not load in stock llama.cpp, LM Studio or Ollama.

| Tier | Size | Quality | Use |
| --- | ---: | --- | --- |
| `IQ2_XXS-GPTQ` | 124.5 GB | assistant-turn KL 0.284, top-1 84.4 % | **Best 2-bit:** sequential GPTQ, 27 % lower KL than plain `IQ2_XXS` |
| `IQ1_M` | 106.6 GB | top-1 47.1 % | Smallest usable tier; 1-bit quality is degraded |
| `IQ1_S` | 94.8 GB | top-1 44.3 % | Do not use: its output is incoherent |
| `IQ2_KT` | 123.8 GB | top-1 56.3 % | Best 2-bit per byte of the plain tiers; Sparsa only |
| `IQ2_XXS` | 124.5 GB | top-1 52.0 % | Also loads in patched llama.cpp |
| `IQ2_XS` | 133.1 GB | not measured | For patched llama.cpp |
| `IQ2_XS-D3` | 146.1 GB | top-1 59.4 % | Fits a 128 GB PC with room to spare |
| `IQ2_KL` | 152.7 GB | top-1 59.8 % | Slightly better than D3; Sparsa only |
| `IQ3_K` | 191.2 GB | top-1 69.8 % | 3-bit; Sparsa only |
| `IQ4_KS` | 233.6 GB | top-1 74.4 % | Closest IQ tier to NVFP4; Sparsa only |
| `NVFP4` | 246.1 GB | top-1 77.3 % | A lossless copy of the checkpoint's 4-bit experts; Sparsa only |

Two measures are used:
- **Assistant-turn KL:** the 4-bit teacher's own answers to 14 held-out prompts, scored with
  `tools/minimax/score_assistant.py`. Lower is better.
- **Top-1:** agreement with an FP32 reference on 6 short prompts.

The two are not comparable with each other. Pick the largest tier that fits in RAM + VRAM with ~10 GB to spare;
Sparsa streams the rest from the SSD.

## Run it

You need an NVIDIA card (12 GB is enough; CUDA 12.8 or newer for an RTX 50 series), RAM for most of the tier, and
Python 3.10+. The speeds above come from the current `main`, so build from source (below) to get them. The last
binary release, [v0.1.3](https://github.com/LyJonathan/Sparsa/releases/latest), is older and slower.

```sh
git clone https://github.com/LyJonathan/Sparsa.git
cd Sparsa
python -m pip install -r requirements.txt
python sparsa.py download iq2-xs-d3 --dir D:/models
python sparsa.py download tokenizer --dir D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3/tokenizer
python sparsa.py download mmproj --dir D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3   # optional, pictures
python sparsa.py serve --model D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3
```

`serve` loads the model (a minute or two). It then serves a chat page and two APIs on `http://127.0.0.1:8080`:
OpenAI's `/v1/chat/completions` and Anthropic's `/v1/messages`, model `minimax-m3.1-flash`, with streaming and tool
calls.

Useful flags: `--access lan` (shares it, with an API key), `--context 8192`, `--gpu 0,1`, `--threads N`, `--ram-gib N`
(cap the RAM tier), `--plan`. `python sparsa.py pick` prints the tier this machine would choose.

## Build from source

Linux (CUDA 12.8+, CMake, Ninja, a C++17 compiler):

```sh
JOBS=8 STRATA_CUDA_ARCHS="75;80;86;89;120" tools/minimax/build_linux.sh
```

Windows (Visual Studio 2022+, CUDA 13.3, Ninja):

```powershell
$env:STRATA_GGML_DIR = 'C:\src\llama.cpp'    # a llama.cpp checkout at third_party/ggml/VERSION.txt
python -m pip install -r requirements-build.txt
powershell -File tools/minimax/build_windows.ps1 -Targets strata-minimax -CudaArch "75;80;86;89;120"
```

`120` is the RTX 50 series and `86` the RTX 30 series; each extra architecture adds build time. Without
`STRATA_GGML_DIR`, CMake fetches the pinned llama.cpp commit. Design notes and every measurement:
[tools/minimax/README.md](tools/minimax/README.md). The quantizer that made the GPTQ tier is in
[tools/minimax/quant/](tools/minimax/quant/).

## Known limitations

- **Decode is bound by memory bandwidth.** The experts that do not fit in VRAM are read from RAM every token, so
  ~8 tokens/s is close to what a dual-channel DDR4 box can feed. More VRAM, faster RAM or a smaller tier raises it.
- **Rare stall.** About 5 % of runs stall in the CPU-expert prompt path of
  [PR #34](https://github.com/LyJonathan/Sparsa/pull/34), so that PR is not merged.
- **Context.** Measured up to 32K. Longer contexts are untested.
- **Vision.** Images only, no video. The encoder runs on the CPU in fp32 and takes 64-80 s per 640x480 image.
- **Tool calls** use MiniMax's XML form and are parsed whole, so their arguments are not streamed.
- **Tested** on Linux with two NVIDIA cards. Windows builds; Windows and single-GPU boxes were not re-measured for
  these numbers.
- **CI** is a Linux build only (no GPU, no model).

## Contributing

PRs that make Sparsa faster or better are welcome: speed, a tier's quality, the open prompt stall. See the
[issues](https://github.com/LyJonathan/Sparsa/issues) and [pull requests](https://github.com/LyJonathan/Sparsa/pulls).
If it runs on your box, a star helps other people find it.

## Upstream Strata

The Qwen side of this repository is upstream's and still works: the `strata` Terminal, the desktop app,
`docs/DETAILS.md`. Upstream's README is kept as [README-STRATA.md](README-STRATA.md), with translated copies beside
it.

## Licence and credits

Sparsa is MIT (`LICENSE`). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the code carried from other
projects. The model files are under the
[MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/LICENSE) and are not part
of this repository.

- [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash): the model, its chat template and its vision
  encoder.
- [Niko1221/Strata](https://github.com/Niko1221/Strata) and contributors: the fork's base (the server, the Terminal,
  the app, the Qwen engine).
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp): the GGUF format, quantization and the CUDA kernels the
  engine reuses (stb_image comes from it too). Its MMQ design informed the expert-GEMM work.
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by Iwan Kawrakow: the IQ2_KT, IQ2_KL, IQ3_K and IQ4_KS
  formats the expert tiers port.
- [Project Maya](https://github.com/mw00/project-maya) (Peasant Smith and contributors, MIT): the GPTQ
  error-feedback rounding the quantizer ports, the Maya-S layout we compare against, and the tier ideas the engine
  measures itself against.
- [agentionai/llama.cpp](https://github.com/agentionai/llama.cpp): the trellis expert types.
- [Strata-DS](https://github.com/Jackwwg83/Strata-DS) (Jackwwg83): the server front end the MiniMax tokenizer
  handling follows.
- [Xolotl](https://github.com/Indras-Mirror/Xolotl) (MIT): the resident-expert routing bias the opt-in route nudge
  follows.
- Unsloth (the dynamic quant-mix idea the low tiers borrow) and ubergarm (ik_llama.cpp quant recipes).
