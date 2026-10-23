#!/usr/bin/env python3
"""Write the MiniMax-M3 vision projector GGUF via llama.cpp.

argv: MODEL_DIR LLAMA_CPP_DIR OUTFILE

Vision weights are BF16. The tower is shard 48 and part of the projector is
shard 13. Exit 2 when either shard is missing so the caller can skip without
inventing a file. Exit 0 when convert_hf_to_gguf.py --mmproj succeeds.
"""

import os
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: mmproj.py MODEL_DIR LLAMA_CPP_DIR OUTFILE", file=sys.stderr)
        return 1
    model, llama, out = sys.argv[1:]
    missing = []
    for name in ("model-00013-of-00048.safetensors", "model-00048-of-00048.safetensors"):
        if not os.path.isfile(os.path.join(model, name)):
            missing.append(name)
    if missing:
        print("SKIP mmproj: missing " + " ".join(missing), file=sys.stderr)
        return 2
    conv = os.path.join(llama, "convert_hf_to_gguf.py")
    if not os.path.isfile(conv):
        print("convert_hf_to_gguf.py not found under " + llama, file=sys.stderr)
        return 1
    cmd = [sys.executable, conv, "--mmproj", "--outtype", "f16", "--outfile", out, model]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
