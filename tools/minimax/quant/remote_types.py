"""Expert tensor types of a GGUF split set on Hugging Face, read from the file headers only (HTTP range requests).

seq_gptq.py needs the template's per-layer types; this gets them without downloading the template itself.

    python remote_types.py LyJonathan/MiniMax-M3.1-Flash-GGUF MiniMax-M3.1-Flash-IQ1_M > iq1m_types.json
"""
import json
import struct
import sys

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Short(Exception):
    pass


class Buf:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, n):
        if self.o + n > len(self.b):
            raise Short
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def str(self):
        return self.take(self.u64()).decode("utf8", "replace")

    def skip_value(self, t):
        if t == 8:
            self.str()
        elif t == 9:
            it, n = self.u32(), self.u64()
            if it == 8:
                for _ in range(n):
                    self.str()
            else:
                self.take(SIZES[it] * n)
        else:
            self.take(SIZES[t])


def tensor_types(blob):
    r = Buf(blob)
    if r.take(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    r.u32()
    nt, nkv = r.u64(), r.u64()
    for _ in range(nkv):
        r.str()
        r.skip_value(r.u32())
    out = {}
    for _ in range(nt):
        name = r.str()
        nd = r.u32()
        r.take(8 * nd)
        out[name] = r.u32()
        r.u64()
    return out


def main():
    repo, folder = sys.argv[1], sys.argv[2]
    files = sorted(f.path for f in HfApi().list_repo_tree(repo, path_in_repo=folder) if f.path.endswith(".gguf"))
    hdr = {"Authorization": f"Bearer {get_token()}"} if get_token() else {}
    types = {}
    for f in files:
        n = 1 << 22
        while True:
            resp = requests.get(hf_hub_url(repo, f), headers={**hdr, "Range": f"bytes=0-{n - 1}"}, timeout=120)
            resp.raise_for_status()
            try:
                tt = tensor_types(resp.content)
                break
            except Short:
                n *= 4
        for name, t in tt.items():
            p = name.split(".")
            if len(p) == 4 and p[2] in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                types[f"{int(p[1])}.{p[2].split('_')[1]}"] = t
        print(f"{f}: {len(tt)} tensors", file=sys.stderr)
    json.dump(types, sys.stdout, indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
