#!/usr/bin/env bash
# Run or prefetch the MM-IQ2_XS split job on this machine.
#   ./run_tier.sh --plan
#   ./run_tier.sh --tier MM-IQ2_XS --machine rig-2 --prefetch --model ~/strata-work/shards
#   ./run_tier.sh --tier MM-IQ2_XS --machine rig-2 --imatrix ~/strata-work/imatrix.gguf \
#       --model ~/strata-work/shards --exe ~/strata-work/build-mm-quant/strata-mm-quant \
#       --out-dir ~/strata-work/gguf
set -euo pipefail
if [[ -f "${HOME}/strata-work/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/strata-work/env.sh"
fi
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
exec "$PY" "$ROOT/run_tier.py" "$@"
