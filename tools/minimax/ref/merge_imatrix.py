"""Merge imatrix partials and write the final llama.cpp GGUF imatrix.

Calibration chunks are independent: for a linear, the per-input-column
``sum(x**2)`` and the row count simply add.  Each machine can therefore run a
disjoint set of chunks (``imatrix.py --shard-index I --shard-count N``) and we
sum the partials here.

Partial files are themselves GGUF imatrix files, so a half-finished machine can
still be merged.  Output format follows spec F.1.1.

Usage:
    python merge_imatrix.py --out imatrix-minimax-m3.1.gguf partial1.gguf partial2.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

STATS_SCHEMA = [
    "sum_sq",
    "mean",
    "elements",
    "std_deviation",
    "skewness",
    "kurtosis",
    "gain",
    "h_norm",
    "l2_dist",
    "cossim",
    "pearson",
    "covariance",
]


def _field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        return f.contents()
    except Exception:
        return default


def read_partial(path: str):
    import gguf

    r = gguf.GGUFReader(str(path))
    datasets = _field(r, "imatrix.datasets", [])
    if isinstance(datasets, str):
        datasets = [datasets]
    chunk_count = int(_field(r, "imatrix.chunk_count", 0) or 0)
    chunk_size = int(_field(r, "imatrix.chunk_size", 0) or 0)
    sums = {}
    counts = {}
    for t in r.tensors:
        name = t.name
        if name.endswith(".in_sum2"):
            base = name[: -len(".in_sum2")]
            sums[base] = np.asarray(t.data, dtype=np.float64)
        elif name.endswith(".counts"):
            base = name[: -len(".counts")]
            counts[base] = np.asarray(t.data, dtype=np.float64).reshape(-1)
        elif name.endswith(".stats") or name.endswith(".in_sum"):
            continue
        else:
            print(f"warning: {path}: ignoring unexpected tensor {name}", file=sys.stderr)
    if set(sums) != set(counts):
        raise ValueError(f"{path}: in_sum2/counts name mismatch")
    return {
        "path": str(path),
        "datasets": list(datasets),
        "chunk_count": chunk_count,
        "chunk_size": chunk_size,
        "sums": sums,
        "counts": counts,
    }


def merge(partials, out, dataset_name=None):
    acc = {}
    ds = []
    chunk_count = 0
    chunk_size = 0
    for p in partials:
        part = read_partial(p)
        ds.extend(part["datasets"])
        chunk_count += part["chunk_count"]
        chunk_size = max(chunk_size, part["chunk_size"])
        for name, s in part["sums"].items():
            if name not in acc:
                acc[name] = {"sum2": s.copy(), "count": part["counts"][name].copy()}
            else:
                if acc[name]["sum2"].shape != s.shape:
                    raise ValueError(f"{name}: shape mismatch {acc[name]['sum2'].shape} vs {s.shape}")
                acc[name]["sum2"] += s
                acc[name]["count"] += part["counts"][name]
        print(f"merged {p}: {len(part['sums'])} entries, {part['chunk_count']} chunks")

    if not acc:
        raise SystemExit("no entries to merge")

    import gguf

    w = gguf.GGUFWriter(str(out), arch="minimax-m3")
    w.add_type("imatrix")
    w.add_array("imatrix.datasets", dataset_name and [dataset_name] or ds or ["merged"])
    w.add_uint32("imatrix.chunk_count", int(chunk_count))
    w.add_uint32("imatrix.chunk_size", int(chunk_size))
    w.add_array("imatrix.stats_schema", STATS_SCHEMA)
    for name in sorted(acc):
        e = acc[name]
        w.add_tensor(name + ".in_sum2", e["sum2"].astype(np.float32))
        w.add_tensor(name + ".counts", e["count"].astype(np.float32).reshape(-1, 1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}: {len(acc)} entries, {chunk_count} chunks")
    return len(acc), chunk_count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("partials", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-name", default=None)
    args = ap.parse_args()
    missing = [p for p in args.partials if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing partials: {missing}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merge(args.partials, args.out, args.dataset_name)


if __name__ == "__main__":
    main()
