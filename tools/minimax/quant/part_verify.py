#!/usr/bin/env python3
"""Check one final split part of a MiniMax-M3.1-Flash GGUF tier before upload.

Reads the GGUF header itself (no ggml, no gguf-py) and reports, in this order:

  header   magic, version, tensor count, key metadata count
  split    split.no / split.count / split.tensors.count against the part asked for
  experts  every routed-expert tensor is the tier's type (ffn_down_exps may be one step up,
           --expert-down), and there are 3 per MoE layer
  size     the file is exactly data_offset + the last tensor's end (nothing truncated, nothing appended)
  sha256   the whole file

Exit code 0 means every check passed. Any failure prints FAIL with the reason and exits 1.

  part_verify.py --gguf FILE --part K --count N --expert IQ3_K [--expert-down IQ1_M] [--json]
"""

import argparse
import hashlib
import json
import os
import struct
import sys

# ggml type id -> (name, block_elems, block_bytes, row_meta).  The ik ids and their geometry are
# include/strata/kernels/iqk_types.hpp; nvfp4 is ggml's GGML_TYPE_NVFP4.
TYPES = {
    0: ("f32", 1, 4, 0),
    1: ("f16", 1, 2, 0),
    2: ("q4_0", 32, 18, 0),
    3: ("q4_1", 32, 20, 0),
    8: ("q8_0", 32, 34, 0),
    14: ("q6_k", 256, 210, 0),
    16: ("iq2_xxs", 256, 66, 0),
    17: ("iq2_xs", 256, 74, 0),
    18: ("iq3_xxs", 256, 98, 0),
    19: ("iq1_s", 256, 50, 0),
    20: ("iq4_nl", 32, 18, 0),
    22: ("iq2_s", 256, 82, 0),
    29: ("iq1_m", 256, 56, 0),
    30: ("bf16", 1, 2, 0),
    40: ("nvfp4", 64, 36, 0),
    160: ("IQ2_KT", 256, 68, 4),
    161: ("IQ3_K", 256, 110, 0),
    162: ("IQ4_KS", 256, 136, 4),
    163: ("IQ3_KT", 256, 100, 4),
    164: ("IQ2_KL", 256, 86, 2),
}

TYPE_ID = {v[0].upper(): k for k, v in TYPES.items()}
TYPE_ID.update({"IQ2_KT": 160, "IQ3_K": 161, "IQ4_KS": 162, "IQ3_KT": 163, "IQ2_KL": 164, "NVFP4": 40})

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def row_bytes(t, n):
    if t not in TYPES:
        return 0
    _name, be, bb, meta = TYPES[t]
    if n % be:
        return 0
    return meta + bb * (n // be)


def need(f, n):
    b = f.read(n)
    if len(b) != n:
        raise SystemExit("short read in %s" % f.name)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--part", type=int, required=True, help="zero-based part index")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--expert", required=True, help="expert type name of the tier, e.g. IQ3_K or NVFP4")
    ap.add_argument("--expert-down", default="",
                    help="type of ffn_down_exps when the recipe steps it up, e.g. IQ1_M; "
                         "default is the same as --expert")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    want_expert = TYPE_ID.get(a.expert.upper())
    if want_expert is None:
        raise SystemExit("unknown expert type " + a.expert)
    want_down = want_expert
    if a.expert_down:
        want_down = TYPE_ID.get(a.expert_down.upper())
        if want_down is None:
            raise SystemExit("unknown down expert type " + a.expert_down)
    fails = []
    info = {"file": a.gguf, "part": a.part, "count": a.count, "expert": a.expert,
            "expert_down": a.expert_down or a.expert}

    with open(a.gguf, "rb") as f:
        magic = need(f, 4)
        if magic != b"GGUF":
            fails.append("bad magic %r" % (magic,))
        version = struct.unpack("<I", need(f, 4))[0]
        n_tensors = struct.unpack("<Q", need(f, 8))[0]
        n_kv = struct.unpack("<Q", need(f, 8))[0]
        info["version"] = version
        info["tensors"] = n_tensors
        info["kv"] = n_kv
        if version != 3:
            fails.append("version %d, expected 3" % version)

        kv = {}
        present = set()
        for _ in range(n_kv):
            klen = struct.unpack("<Q", need(f, 8))[0]
            key = need(f, klen).decode("utf-8", "replace")
            present.add(key)
            typ = struct.unpack("<I", need(f, 4))[0]
            if typ == 8:
                vlen = struct.unpack("<Q", need(f, 8))[0]
                raw = need(f, vlen)
                if key in ("split.no", "split.count", "split.tensors.count", "general.architecture"):
                    kv[key] = raw.decode("utf-8", "replace")
            elif typ == 9:
                et = struct.unpack("<I", need(f, 4))[0]
                cnt = struct.unpack("<Q", need(f, 8))[0]
                if et == 8:
                    for _ in range(cnt):
                        slen = struct.unpack("<Q", need(f, 8))[0]
                        need(f, slen)
                else:
                    need(f, SCALAR.get(et, 4) * cnt)
                kv[key] = "<array>"
            elif typ == 4:
                kv[key] = struct.unpack("<I", need(f, 4))[0]
            elif typ == 5:
                kv[key] = struct.unpack("<i", need(f, 4))[0]
            elif typ == 2:
                kv[key] = struct.unpack("<H", need(f, 2))[0]
            elif typ in (0, 1, 3, 6, 7, 10, 11, 12):
                need(f, SCALAR[typ])
                kv[key] = "<scalar>"
            else:
                fails.append("unknown kv type %d for %s" % (typ, key))
                break

        info["split_no"] = kv.get("split.no", "ABSENT")
        info["split_count"] = kv.get("split.count", "ABSENT")
        info["split_tensors_count"] = kv.get("split.tensors.count", "ABSENT")
        info["architecture"] = kv.get("general.architecture", "ABSENT")
        info["alignment"] = kv.get("general.alignment", 32)

        if a.part == 0:
            if kv.get("general.architecture") != "minimax-m3":
                fails.append("general.architecture is %r" % (kv.get("general.architecture"),))
            # the vocabulary and the chat template are the part-0 markers
            for want in ("tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
                if want not in present:
                    fails.append("part 0 has no " + want)
        else:
            # later parts carry only split metadata; a stray architecture key would mean the
            # globals were written twice
            if "general.architecture" in present:
                fails.append("part %d repeats general.architecture" % a.part)
        if kv.get("split.no") != a.part:
            fails.append("split.no is %r, expected %d" % (kv.get("split.no"), a.part))
        if kv.get("split.count") != a.count:
            fails.append("split.count is %r, expected %d" % (kv.get("split.count"), a.count))
        if "split.tensors.count" not in kv:
            fails.append("split.tensors.count is absent")

        tensors = []
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", need(f, 8))[0]
            name = need(f, nlen).decode("utf-8", "replace")
            nd = struct.unpack("<I", need(f, 4))[0]
            dims = struct.unpack("<%dQ" % nd, need(f, 8 * nd))
            typ = struct.unpack("<I", need(f, 4))[0]
            off = struct.unpack("<Q", need(f, 8))[0]
            n = int(dims[0])
            rb = row_bytes(typ, n)
            if rb == 0:
                fails.append("no row size for %s type %d at %d" % (name, typ, n))
                continue
            ne = 1
            for d in dims[1:]:
                ne *= int(d)
            tensors.append((name, typ, off, rb * ne))
        header_end = f.tell()

    align = int(info["alignment"] or 32)
    data_start = (header_end + align - 1) // align * align
    end = max((off + nb for _n, _t, off, nb in tensors), default=0)
    expected = data_start + end
    actual = os.path.getsize(a.gguf)
    info["size"] = actual
    info["size_expected"] = expected
    if actual != expected:
        fails.append("size %d, header says %d" % (actual, expected))

    expert_names = [t for t in tensors if t[0].endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight",
                                                         "ffn_down_exps.weight"))]
    bad = [(n, t) for n, t, _o, _b in expert_names
           if t != (want_down if n.endswith("ffn_down_exps.weight") else want_expert)]
    info["expert_tensors"] = len(expert_names)
    info["expert_types"] = sorted({TYPES.get(t, (str(t),))[0] for _n, t, _o, _b in expert_names})
    if bad:
        fails.append("%d expert tensors are not %s/%s: %s" % (len(bad), a.expert, info["expert_down"], bad[:3]))
    if not expert_names:
        fails.append("no expert tensors in this part")

    h = hashlib.sha256()
    with open(a.gguf, "rb") as f:
        while True:
            b = f.read(32 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    info["sha256"] = h.hexdigest()

    if a.json:
        print(json.dumps(info))
    else:
        print("header  ok  version=%d tensors=%d kv=%d arch=%s" %
              (info.get("version", 0), info["tensors"], info["kv"], info.get("architecture")))
        print("split   %s  no=%r count=%r tensors.count=%r" %
              ("ok" if not [x for x in fails if x.startswith("split")] else "FAIL",
               info["split_no"], info["split_count"], info["split_tensors_count"]))
        print("experts %s  %d tensors  types=%s  want=%s/%s" %
              ("ok" if not bad else "FAIL", len(expert_names), ",".join(info["expert_types"]),
               a.expert, info["expert_down"]))
        print("size    %s  %d bytes (header %d)" %
              ("ok" if actual == expected else "FAIL", actual, expected))
        print("sha256  %s" % info["sha256"])
    if fails:
        for x in fails:
            print("FAIL " + x, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
