"""Write seq_gptq.py's per-layer expert bytes into a GGUF split set, in place.

The bytes must have exactly the size of the tensor they replace (seq_gptq.py takes the template's types), so this only
overwrites tensor data; metadata and every other tensor stay as they are. Run it on a *copy* of the template.

    python splice_experts.py --gguf <copy, part 1> --bytes D:/mm-ref/v2 [--layers 3-59]
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "_baseline" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--bytes", required=True)
    ap.add_argument("--layers", default="0-59")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-"))
    first = pathlib.Path(a.gguf)
    stem = first.name.split("-00001-of-")[0]
    src = pathlib.Path(a.bytes)
    done = 0
    for f in sorted(first.parent.glob(stem + "-0*-of-*.gguf")):
        jobs = []
        for t in gguf.GGUFReader(str(f)).tensors:
            p = t.name.split(".")
            if len(p) == 4 and p[2] in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps") and lo <= int(p[1]) <= hi:
                b = src / f"L{int(p[1]):02d}_{p[2].split('_')[1]}.bin"
                if not b.exists():
                    continue
                if b.stat().st_size != t.n_bytes:
                    raise SystemExit(f"{t.name}: {b} is {b.stat().st_size} bytes, tensor is {t.n_bytes}")
                jobs.append((t.data_offset, b, t.name))
        if not jobs:
            continue
        with open(f, "r+b") as fh:
            for off, b, name in jobs:
                fh.seek(off)
                with open(b, "rb") as bh:
                    while chunk := bh.read(1 << 26):
                        fh.write(chunk)
                done += 1
                print(f"{f.name}: {name}", flush=True)
    print(f"spliced {done} tensors")


if __name__ == "__main__":
    main()
