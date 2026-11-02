#!/bin/bash
STRATA_IQ_MT=0 exec python tools/minimax/pc_launch.py --owner dspark -- ./build/strata-minimax.exe "$@"
