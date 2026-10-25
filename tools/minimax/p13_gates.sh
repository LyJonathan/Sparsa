#!/bin/bash
# tools/minimax/p13_gates.sh - the P13 gates, in order, on the slice (they need a free PC: pc-engine.lock = dspark).
#
#   1. the multi-token CPU row equals ggml's one-pair dot, bit for bit (iq_mt_test, 0 mismatches)
#   2. a window (8 rows) equals the token path on the same placement: gate_logits, max abs 0.0
#   3. the same with STRATA_IQ_MT=0 (the multi-token row changes nothing at all)
#   4. 200-token prefill, STRATA_IQ_MT=1 against =0
#   5. greedy decode: plain vs DSpark, and every draft rejected (the rollback arm)
#
# Everything runs on the 15-layer real-weight slice with all 128 experts a layer pinned in RAM, so every read is from
# memory and the expert kernel is the same on both sides.
set -u
cd "$(dirname "$0")/../.." || exit 1
M="../models/quant-d3/p5a-l0-14-iq2xs.gguf"
PACK="../models/quant-d3/dspark-slice-q4_0.gguf"
BASE_ARGS="--gpus 0 --minimax-dense-layers-gpu 15 --minimax-tiers --minimax-vram-slots-per-layer 0 \
--minimax-ram-experts-per-layer 128 --minimax-kv vram --cpu-threads 12 --memory-limit-mib 40000 \
--minimax-allow-nonstandard"
EXE=./build/strata-minimax.exe
# Every engine run goes through the shared-PC launcher (lock, RAM floor, watchdog).  gate_logits.py takes a single
# `--exe` string, so it is handed a shim that execs the launcher around the engine.
OWNER=${PC_OWNER:-dspark}
LAUNCH=tools/minimax/pc_launch.py
if [ ! -f "$LAUNCH" ]; then echo "pc_launch.py is missing: run nothing on the shared PC without it" >&2; exit 1; fi
# gate_logits.py --exe takes a whole command (shlex), so the launcher goes in front of the engine there too.
# EST_RAM_GB overrides the estimate; by default the launcher reads the engine's own flags.
if [ -n "${EST_RAM_GB:-}" ]; then EST="--est-ram-gb $EST_RAM_GB "; else EST=""; fi
SHIM="python $LAUNCH --owner $OWNER ${EST}-- $EXE"
TMPD=$(mktemp -d)
IDS=tools/minimax/p13_ids.txt
CTX=1024

echo "=== 1. the multi-token CPU row vs ggml's one-pair dot ==="
./build/iq_mt_test.exe --rows 300 || exit 1

python - <<'PY'
from tokenizers import Tokenizer
t = Tokenizer.from_file('D:/hf-d3/IQ2_XS-D3/tokenizer/tokenizer.json')
text = ("The history of the printing press begins in the fifteenth century, when movable type made it possible to "
        "reproduce a page of text quickly and cheaply. Within fifty years, presses were operating in more than two "
        "hundred towns across Europe, and the number of books in circulation had grown from a few thousand to "
        "millions. Printers experimented with typefaces, page layouts and paper sizes, and each experiment changed how "
        "readers encountered a text.")
ids = t.encode(text).ids * 3   # ~250 tokens: the 200-token prefill A/B takes the first 200
open('tools/minimax/p13_ids.txt', 'w').write(' '.join(map(str, ids)))
PY

echo
echo "=== 2. a window (8 rows) vs the token path, same placement ==="
python tools/minimax/gate_logits.py --exe "$SHIM" --exe "$SHIM" --label token --label window \
  --model $M --ids $IDS --ctx $CTX --allow \
  --arm-args "$BASE_ARGS --minimax-prefill-batch 0" \
  --arm-args "$BASE_ARGS --minimax-prefill-batch 8 --minimax-prefill-experts cpu"

echo
echo "=== 3. control: the multi-token row off (STRATA_IQ_MT=0) ==="
STRATA_IQ_MT=0 python tools/minimax/gate_logits.py --exe "$SHIM" --exe "$SHIM" --label token --label window-noMT \
  --model $M --ids $IDS --ctx $CTX --allow \
  --arm-args "$BASE_ARGS --minimax-prefill-batch 0" \
  --arm-args "$BASE_ARGS --minimax-prefill-batch 8 --minimax-prefill-experts cpu"

echo
echo "=== 4. 200-token prefill, multi-token row on and off ==="
for rep in 1 2 3; do
  for mt in 1 0; do
    STRATA_IQ_MT=$mt timeout 900 python $LAUNCH --owner $OWNER ${EST}-- $EXE --model $M --serve $BASE_ARGS --max-context $CTX \
      --minimax-prefill-batch 200 --minimax-prefill-experts cpu <<< "GEN 0 temperature=0 $(cut -d" " -f1-200 $IDS)
STOP" 2>/dev/null | awk -v mt=$mt '/^DONE /{printf "rep %s mt=%s prompt_ms=%s (%d tokens) pp=%.1f tok/s\n", "'"$rep"'", mt, $4, $3, $3/$4*1000}'
  done
done

echo
echo "=== 5. greedy decode: plain, DSpark, and every draft rejected ==="
run() {
  local label=$1; shift
  local env=""
  if [ "$label" = "corrupt" ]; then env="STRATA_DSPARK_CORRUPT=1"; fi
  env $env timeout 900 python $LAUNCH --owner $OWNER ${EST}-- $EXE --model $M --serve $BASE_ARGS --max-context $CTX --minimax-prefill-batch 512 "$@" \
    <<< "GEN 32 temperature=0 seed=1 $(cut -d" " -f1-60 $IDS)
STOP" 2>$TMPD/p13.$label.err | grep -E "^T " | awk '{printf "%s ", $2}' > $TMPD/p13.$label.toks
  echo "  $label: $(cat /tmp/p13.$label.toks)"
  grep -E "INFO minimax-spec" $TMPD/p13.$label.err | tail -1 | sed 's/^/  /'
}
run plain
run dspark --dspark-model $PACK --speculative 4
run corrupt --dspark-model $PACK --speculative 4
TMPD=$TMPD python - <<'PY'
import os, pathlib
td = pathlib.Path(os.environ['TMPD'])
p = (td / 'p13.plain.toks').read_text().split()
d = (td / 'p13.dspark.toks').read_text().split()
c = (td / 'p13.corrupt.toks').read_text().split()
print(f'  plain {len(p)} tokens, dspark {len(d)}, corrupt {len(c)}')
print(f'  dspark identical to plain: {p == d}')
print(f'  corrupt (every draft rejected, every round rolled back) identical to plain: {p == c}')
if p != d:
    i = next((k for k in range(min(len(p), len(d))) if p[k] != d[k]), None)
    print(f'  first difference at token {i}')
PY
