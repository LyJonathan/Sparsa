#!/usr/bin/env python3
"""Compare GGUF tensor payloads by name.

llama-gguf-split --merge rewrites the header (split.count becomes 0, and ggml
serializes the container). Payload hashes are the bit-identity check. A whole
file hash is printed too.
"""

import hashlib
import struct
import sys

# ggml type -> (block size, block bytes). Enough for the Q8_0 split test and IQ/Q types.
TYPE_SIZE = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 36),     # Q8_1
    10: (256, 84),    # Q2_K
    11: (256, 110),   # Q3_K
    12: (256, 144),   # Q4_K
    13: (256, 176),   # Q5_K
    14: (256, 210),   # Q6_K
    15: (256, 292),   # Q8_K
    16: (256, 66),    # IQ2_XXS
    17: (256, 74),    # IQ2_XS
    18: (256, 98),    # IQ3_XXS
    19: (256, 50),    # IQ1_S
    20: (256, 56),    # IQ1_M
    22: (32, 18),     # IQ4_NL
    23: (256, 110),   # IQ3_S
    24: (256, 82),    # IQ2_S
    25: (256, 136),   # IQ4_XS
}


def u32(f):
    return struct.unpack("<I", f.read(4))[0]


def u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_string(f):
    n = u64(f)
    data = f.read(n)
    if len(data) != n:
        raise SystemExit("short string in " + f.name)
    return data


def skip_value(f, typ):
    widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if typ in widths:
        f.seek(widths[typ], 1)
        return
    if typ == 8:
        f.seek(u64(f), 1)
        return
    if typ == 9:
        elem = u32(f)
        count = u64(f)
        for _ in range(count):
            skip_value(f, elem)
        return
    raise SystemExit("unsupported GGUF value type %d" % typ)


def tensor_nbytes(typ, ne):
    if typ not in TYPE_SIZE:
        raise SystemExit("no size for ggml type %d" % typ)
    blck, bsize = TYPE_SIZE[typ]
    if ne[0] % blck != 0:
        raise SystemExit("row %d is not a multiple of block %d" % (ne[0], blck))
    row = (ne[0] // blck) * bsize
    n = row
    for dim in ne[1:]:
        n *= dim
    return n


def load(path):
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        raise SystemExit(path + " is not GGUF")
    version = u32(f)
    if version != 3:
        raise SystemExit(path + " is GGUF version %d" % version)
    n_tensors = u64(f)
    n_kv = u64(f)
    for _ in range(n_kv):
        read_string(f)
        skip_value(f, u32(f))
    tensors = []
    for _ in range(n_tensors):
        name = read_string(f).decode("utf-8")
        n_dims = u32(f)
        ne = [u64(f) for _ in range(n_dims)]
        while len(ne) < 4:
            ne.append(1)
        typ = u32(f)
        offset = u64(f)
        tensors.append((name, typ, ne, offset, tensor_nbytes(typ, ne)))
    pos = f.tell()
    data = (pos + 31) & ~31
    out = {}
    for name, typ, ne, offset, nbytes in tensors:
        if name in out:
            raise SystemExit("duplicate tensor " + name)
        f.seek(data + offset)
        digest = hashlib.sha256()
        left = nbytes
        while left:
            buf = f.read(min(left, 1 << 20))
            if not buf:
                raise SystemExit("short payload for " + name)
            digest.update(buf)
            left -= len(buf)
        out[name] = (digest.hexdigest(), nbytes, typ)
    f.close()
    return out


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def main():
    if len(sys.argv) != 3:
        print("usage: compare_gguf.py A.gguf B.gguf", file=sys.stderr)
        return 2
    a_path, b_path = sys.argv[1], sys.argv[2]
    a = load(a_path)
    b = load(b_path)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    differ = []
    for name in sorted(set(a) & set(b)):
        if a[name] != b[name]:
            differ.append(name)
    print("tensors A %d  B %d  only-A %d  only-B %d  payload-differ %d" % (len(a), len(b), len(only_a), len(only_b), len(differ)))
    for name in (only_a + only_b + differ)[:30]:
        print(" ", name)
    print("file-sha A", file_sha(a_path))
    print("file-sha B", file_sha(b_path))
    if only_a or only_b or differ:
        return 1
    print("tensor payloads match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
