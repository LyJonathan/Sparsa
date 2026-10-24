# Sparsa

**EXPERIMENTAL.** This is a working preview, not a finished product. It loads and answers, it is slow,
and some parts are not done (below). Expect bugs, and expect the settings to change.

Sparsa is a fork of [Strata](https://github.com/Niko1221/Strata) that adds a runtime for
**MiniMax-M3.1-Flash**: a 427-billion-parameter mixture-of-experts model with 60 layers, 57 of them
holding 128 experts each, of which four run per token. Sparsa runs it on a normal PC: one or two
NVIDIA graphics cards, the system RAM, and the SSD for whatever does not fit.

It is a real fork, not a wrapper: `src/minimax/` is a separate engine (`strata-minimax`) with
MiniMax's sparse attention, its router and shared expert, the expert tiers, and a serve protocol the
Python server speaks. The server is Strata's: OpenAI- and Anthropic-compatible APIs, streaming, tool
calls, a web page.

## Which files it loads

All tiers are on Hugging Face in
[LyJonathan/MiniMax-M3.1-Flash-GGUF](https://huggingface.co/LyJonathan/MiniMax-M3.1-Flash-GGUF),
13 parts each. Sizes are the published bytes; bits/weight is over the 426.2 B language-model
parameters. The measured box below is 2x RTX 5070 12 GB, 128 GB DDR4, one NVMe.

| folder | size | bits/weight | loads in | quality |
| --- | ---: | ---: | --- | --- |
| `MiniMax-M3.1-Flash-IQ2_XS` | 133.1 GB | 2.50 | llama.cpp (patched), Sparsa | KL64 1.92, top-1 46.8 % ¹ |
| `MiniMax-M3.1-Flash-IQ2_XS-D3` | 146.1 GB | 2.74 | llama.cpp (patched), Sparsa | KL64 1.211, top-1 59.4 % |
| `MiniMax-M3.1-Flash-IQ3_K` | 191.2 GB | 3.59 | **Sparsa only** | KL64 0.847, top-1 69.8 % |
| `MiniMax-M3.1-Flash-IQ4_KS` | 233.6 GB | 4.38 | **Sparsa only** | KL64 0.760, top-1 74.4 % |
| `MiniMax-M3.1-Flash-IQ2_KL` | 152.7 GB | 2.87 | **Sparsa only** | KL64 1.184, top-1 59.8 % |
| `MiniMax-M3.1-Flash-NVFP4` | 246.1 GB | 4.62 | **Sparsa only** | not measured |

`Sparsa only` means the expert types (ik_llama.cpp's IQ3_K / IQ4_KS / IQ2_KL, and the checkpoint's
NVFP4 values) are Strata's own type ids; upstream llama.cpp cannot read those files. IQ3_K, IQ4_KS
and IQ2_KL were through the G3 gate on 2026-10-24 (numbers below); NVFP4 has not been measured.

¹ The IQ2_XS row is patched llama.cpp on the same file, 935 positions (the six short prompts plus the
two long ones). The unmarked rows are Sparsa's own G3 gate over the 679 positions of the six short
prompts.

The `IQ1_M` folder on the Hub is incomplete (2 of 13 parts) and cannot be loaded.

## Quickstart

You need: an NVIDIA card (CUDA 12.8 or newer for an RTX 50 series; 12 GB is enough), the RAM the tier
does not fit in, and Python 3.10+.

**1. Get the runtime.** Either unzip a release into a checkout of this repository so the engine sits
in `bin/`, or build it (below). The release has `strata-minimax.exe` (Windows) or
`strata-minimax` (Linux) plus the CUDA libraries it needs.

```sh
git clone https://github.com/LyJonathan/Sparsa.git
cd Sparsa
# put the release's files in bin/: bin/strata-minimax.exe (+ DLLs) or bin/strata-minimax
python -m pip install -r requirements.txt
```

**2. Download a model and the tokenizer.** The GGUF carries the token ids but not the BPE merges, so
the small tokenizer files come from the checkpoint (about 10 MB):

```sh
python sparsa.py download iq2-xs-d3 --dir D:/models     # prints the exact hf command if hf is absent
python sparsa.py download tokenizer --dir D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3/tokenizer
```

`sparsa.py download` uses `hf` (`python -m pip install -U huggingface_hub`). Without it, the same
command is:

```sh
hf download LyJonathan/MiniMax-M3.1-Flash-GGUF \
  --include "MiniMax-M3.1-Flash-IQ2_XS-D3/*" --local-dir D:/models
hf download MiniMaxAI/MiniMax-M3.1-Flash tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3/tokenizer
```

**3. Serve it.**

```sh
python sparsa.py serve --model D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3
```

It prints the run settings it wrote, starts the engine (a minute or two to load), then:

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

Useful flags: `--context 8192` (default 32768; long context is work in progress), `--gpu 0` or
`--gpu 0,1`, `--ram-gib N` to cap the RAM expert tier, `--threads N`, `--port`, `--plan` to print the
settings without starting, `--engine PATH`. `python sparsa.py inspect --model FILE` lists what the
engine reads from a GGUF.

### Share it on your network

By default the server listens on `127.0.0.1` only: this computer. To let another machine on your
network use it, say so, and Sparsa makes an API key and binds every address:

```sh
python sparsa.py serve --model D:/models/MiniMax-M3.1-Flash-IQ2_XS-D3 --access lan
# sparsa: LAN address http://192.168.1.50:8080/v1   (API key: sk-strata-...)
# sparsa: Windows may ask whether to allow Strata (python.exe) on your network: choose Allow...
```

Then from the other machine, the same request with the key:

```sh
curl http://192.168.1.50:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer sk-strata-..." -d '{
  "model": "minimax-m3.1-flash", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 200 }'
```

Anthropic clients send the key as `x-api-key` instead. The key is kept in the run settings file
(`sparsa-minimax.json`, which `.gitignore` covers); `--show-key` prints it again, and
`--no-api-key` serves without one, with a warning: anyone who can reach the port could then use the
model and spend your electricity. Keep the server on your own network; there is no user account, just
the one key.

The server is the Qwen families' server too. `strata` (the Terminal) still manages those;
`python sparsa.py` is the MiniMax path.

### Build from source

Windows (Visual Studio 2022+, CUDA 13.3, Ninja):

```powershell
$env:STRATA_GGML_DIR = 'C:\src\llama.cpp'    # a llama.cpp checkout at third_party/ggml/VERSION.txt
python -m pip install -r requirements-build.txt
powershell -File tools/minimax/build_windows.ps1 -Targets strata-minimax -CudaArch "75;80;86;89;120"
# the worktree must not sit under a path with an apostrophe: nvcc drops those include paths, and
# the script maps the tree to a drive letter (-SubstDrive M: -GgmlSubstDrive N:) for that
```

Linux (CUDA 12.8+, CMake, Ninja, a C++17 compiler):

```sh
JOBS=16 STRATA_CUDA_ARCHS="75;80;86;89;120" tools/minimax/build_linux.sh
```

`-CudaArch`/`STRATA_CUDA_ARCHS` are the card architectures: `120` for the RTX 50 series, `86` for an
RTX 30 series, and so on; a range costs build time per architecture. Without `STRATA_GGML_DIR` CMake
fetches the pinned llama.cpp commit (network needed). Details and every engine flag:
[tools/minimax/README.md](tools/minimax/README.md).

## Hardware

- **Graphics cards.** One or two NVIDIA cards, CUDA compute 7.5 (Turing) or newer. Two cards split the
  layers by free VRAM (`--layer-split auto`); unequal cards work. AMD and Apple are not supported by
  this engine yet.
- **RAM.** The experts that are not in VRAM live in pinned RAM, and the rest stream from the SSD at
  0.9-2 GB/s. 128 GB was the measured box; less RAM works but more of every token comes off the SSD.
- **Disk.** The tier you download, plus about 8 GB of scratch.

## Measured, not estimated

Full `MM-IQ2_XS-D3` (146.1 GB), 2x RTX 5070 12 GB, Ryzen 9 5900X, 128 GB DDR4, Windows, NVMe, this
release, `python sparsa.py serve` with its defaults (one 4096-token prompt chunk, the int8 expert dots, the
RAM tier capped at 84 GiB), the box idle, real text from the quality set:

| what | result | drive read rate during the run |
| --- | ---: | ---: |
| **decode**, 753-token prompt, 64 greedy tokens | **4.17 tok/s** (240 ms/token) | 891 MB/s mean, 1.74 GB/s peak |
| prefill, 411-token chat prompt (time to first token) | 30.5 s = **13.5 tok/s** | 0.9-1.7 GB/s |
| prefill, 753 tokens | 33.8 s = **22.3 tok/s** | 891 MB/s mean |
| prefill, 4,686 tokens | 112.9 s = **41.5 tok/s** | 0.9-1.7 GB/s |
| prefill, 32 K tokens (P11 measurement, same box) | 739.6 s = 44.3 tok/s | 4.4-5.6 GB/s |

The NVMe on this box swings between 0.12 and 5.6 GB/s seen; every number above is a drive-bound number, so it
moves with the drive. The same 753-token prompt took 33.8 s in one run and 59.9 s in another with nothing else
running, so read the prefill rows as "about half a minute for a chat message, a minute or two for a few thousand
tokens", not as a fixed rate.

**Chat reads its prompt in one batched chunk.** That is what the 13.5-22 tok/s rows are: the same 411-token
message through the per-token path (`--prefill-batch 0`) takes **124 s** (3.3 tok/s) because it re-reads every
routed expert once per token. Batching is the 4x, and it is the default.

**About 4.2 tokens a second decoding.** That is experimental and slow. Decode is bound by the experts that are
not in the RAM tier: 7.8 experts per token come off the SSD, and the tier holds 87 of the 128 experts per layer
at 84 GiB (`--ram-gib` raises it if the box has the memory).

Quality, `MM-IQ2_XS-D3` against an fp32 reference of the checkpoint's own weights, six prompts,
679 positions, teacher-forced, the experts projected on the card (the server's path):

| metric | llama.cpp | Sparsa |
| --- | ---: | ---: |
| top-1 agreement | 59.20 % | 60.82 % |
| KL64 mean | 1.3446 | 1.2346 |
| full-vocab KL mean | - | 1.4409 |

The engine matches llama.cpp on the same file, and the earlier CPU-expert arm measured KL64 1.2110 /
top-1 59.35 %, so the placement of the expert projection is not what the residual is. Both sit at KL
~0.7-1.0 where the reference is confident: the residual is the 2-bit expert tier, not the engine.

The same gate rejects the tensor-core expert GEMM (`--minimax-prefill-expert-gemm tc`): KL64 1.2998,
full-vocab KL 1.4886, top-1 58.17 %, and one prompt (code_lru) drops from 38.6 % to 30.7 % top-1. It is
1.33x on a 4 K chunk, so it is kept as an option and is not the default. The slice parity run (a 15-layer
file, IQ2_XS gate/up + Q2_0 down) passed at KL 1.34e-4, which points at the iq3_xxs down projection of
this tier rather than at the idea; that is the next thing to check there.

Quality across the tiers, G3 vs the same fp32 reference, the same six short prompts (679 positions),
teacher-forced:

| tier | KL64 mean | full-vocab KL | top-1 agreement |
| --- | ---: | ---: | ---: |
| `MM-IQ2_XS-D3` | 1.211 | 1.472 | 59.4 % |
| `MM-IQ2_KL` | 1.184 | 1.387 | 59.8 % |
| `MM-IQ3_K` | 0.847 | 0.982 | 69.8 % |
| `MM-IQ4_KS` | 0.760 | 0.905 | 74.4 % |

KL64 is the KL over the reference's top-64 tokens, full-vocab KL over all 200,064. The IQ2_XS-D3 row
was measured on the Windows box above; the three ik_llama tiers on rig-2 (Ryzen 9 5900X, 121 GB RAM,
RTX 4060 Ti + RTX 5060 Ti, Linux) with the experts on the CPU. The gain from 2-bit to 3-bit is real
(59 % to 70 % top-1); from 3-bit to 4-bit it is another 4.6 points.

## Known limits

- **Speed.** 3.3-4.2 tokens a second on the measured box. A 200-400 token chat message takes about half
  a minute before the first token; a few thousand tokens take a minute or two.
- **Long context.** The model supports 1M tokens; this engine has a working 8K-32K path and a long
  context rework in progress. Above 32K expect it to be slow or to fail.
- **Vision.** Not implemented. The image tokens exist in the tokenizer; the encoder does not.
- **Tool calls** work through the XML form MiniMax's template uses, and are parsed whole (arguments
  are not streamed as fragments).
- **Sampling** is the card's (temperature 1.0, top_p 0.95) unless the request says otherwise.
- **NVFP4** has not been through the quality gate. It runs; its quality is unmeasured.
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
- MiniMax - the model and its chat template.
