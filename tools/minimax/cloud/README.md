# G3 on a rented Linux box (molab / Modal sandbox)

The point: a second G3 runner so rig-2 can build quants. The box here was a marimo notebook sandbox on
molab (Modal, gVisor): Debian 13, one RTX PRO 6000 Blackwell (96 GB, sm_120, driver 595.71), CUDA 13.3
from pip, 20 vCPU, a 160 GiB cgroup memory limit and an overlay root filesystem.

## What the box costs you in setup

* **No cmake/ninja**: `python3 -m pip install cmake ninja`, then use `/tmp/uv-venv/bin/cmake`.
  `cmake 4.4.4` and ggml's CMake both work at that version.
* **The nvcc is the pip one**, at `/usr/local/lib/python3.13/site-packages/nvidia/cu13`. Two fixes before
  CMake will configure against it (both in `tools/minimax/build_linux_g3.sh`):
  `lib/libcudart.so` only exists as `libcudart.so.13`, so symlink it; and `include/cccl` is empty while
  `cuda_fp16.h` includes `<nv/target>`, so sparse-checkout `NVIDIA/cccl` and link `nv cuda thrust cub` in.
  Without those two, configure dies at `Could NOT find CUDAToolkit (missing: CUDA_CUDART)` and then at
  `nv/target: No such file or directory`.
* **The filesystem is not RAM-backed**, even though `df` reports 8 EiB. A 20 GiB write left the cgroup
  usage flat at 0.2 GiB and a 400 GiB write succeeded before being deleted, at ~1.8 GiB/s write and
  ~7 GiB/s read-back. So the whole tier fits: there is no memory-pressure reason to chunk the download.
* **HF pull is fast**: `hf download --max-workers 8` moved IQ2_KL (142.2 GiB) in 302 s, ~480 MiB/s.
* **Getting files onto the box** without a token: the repo is public now, so `git clone` for the source;
  for the FP32 reference, push a tar.gz through the notebook's `/api/kernel/execute` endpoint in ~8 MB
  base64 chunks. 497 MB of reference took 173 s that way (~2.9 MB/s). Do not put a GitHub or HF token
  on the box - none is needed.

## Reference data

Only `logits_last128.npy` and the `ids/*.bin` files are needed; `final_hidden.npy` (the big sibling in
`ref_out/logits`) is not read by `g3_sparsa.py`. The 6 short prompts plus `ppl_text_hf.bin` are 519 MiB
raw / 463 MiB gzipped, against 970 MiB for the whole `g3ref` on rig-2.

```
g3ref/logits/<prompt>/logits_last128.npy
g3ref/ids/<prompt>.bin, g3ref/ids/ppl_text_hf.bin
```

## Build

```
CUDA_HOME=/usr/local/lib/python3.13/site-packages/nvidia/cu13 \
STRATA_GGML_DIR=/work/llama.cpp STRATA_BUILD_DIR=/work/build STRATA_CUDA_ARCHS=120 JOBS=20 \
  sh tools/minimax/build_linux_g3.sh
```

llama.cpp is the pin in `third_party/ggml/VERSION.txt` (3cf03257); fetch it shallow:

```
git init llama.cpp && cd llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp.git &&
git fetch --depth 1 origin 3cf03257f219afbe7334045ff7c6a06ac68c627d && git checkout FETCH_HEAD
```

20 vCPU build `strata-minimax` in ~2 minutes. `STRATA_BUILD_TESTS=OFF`; archs `120` only, since the box
has no other card.

## Run

```
TIER=IQ1_M python3 tools/minimax/cloud/run_tier_g3.py --tier $TIER --work /work --exe /work/build/strata-minimax
```

Flags are rig-2's, minus the second GPU and with the box's thread count:

```
--minimax-tiers --gpus 0 --layer-split auto --minimax-kv vram --max-context 8192
--minimax-dense-layers-gpu -1 --minimax-prefill-batch 512 --minimax-prefill-experts cpu
--minimax-prefill-dense f32 --minimax-vram-slots-per-layer 0 --vram-reserve-mib 700
--cpu-threads 20 --memory-limit-mib 140000 --no-ppl
```

`--memory-limit-mib 140000` is the RAM tier: the cgroup is 160 GiB and the kernel plus page cache need
the rest. On this box that is 128 experts per layer resident, so a 142 GiB tier reads nothing from the
SSD tier during a run (`ssd_reads=0`).

`--minimax-prefill-experts gpu` fails on type 164 (IQ2_KL) as it does on rig-2. `--prefill-batch 0` is
never valid.

## Two things that are not what they look like

* `--minimax-vram-slots-per-layer 0` **disables** the VRAM expert tier; it is not "auto". The rig-2 table
  was measured that way (its log line says `vram_slots_per_layer=0`), so leave it at 0 to compare against
  it. Leaving the flag off makes the engine auto-size the tier and, on this box, `cudaMalloc` fails with
  out of memory before `READY`; a hand-set 16 slots/layer does run (37% of expert hits served from VRAM)
  but measured the same decode rate as 0 slots, so the tier was not what limited this box.
* **A tier's numbers depend on the tier composition.** Re-running IQ2_KL here with the same flags gave
  KL64 1.1175 overall against rig-2's 1.1843 on the same bytes (top-1 60.24% vs 59.79%); per prompt the
  spread reaches 13% on `tool_weather` while `math_bakery` agrees to 4e-4. Different RAM-tier size and a
  one-GPU layer split change the prefill batching, and the batched CPU expert path is not bit-identical
  across batch shapes. Compare tiers measured on the same box; use the cross-machine agreement as a
  sanity band (KL64 within ~6%, top-1 within ~0.5 points), not as a bitwise check.
