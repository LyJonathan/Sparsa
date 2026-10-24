"""Package the MiniMax engine the way the releases carry it: `bin/` with the engine and the libraries it needs.

    python tools/minimax/package_release.py --version 0.1.2            # Windows (this machine's build)
    python tools/minimax/package_release.py --version 0.1.2 --sums     # rewrite SHA256SUMS.txt over dist/*

Windows: the archive holds `bin/strata-minimax.exe`, the CUDA runtime it imports (cudart, cublas, cublasLt,
the DLL *closure*, not the whole toolkit) and the VC++ runtime.  The engine looks for its libraries beside
the executable, so the folder works from an empty directory with no CUDA toolkit installed (only the driver).

Linux: run the same command on the build machine with `--platform linux`; it copies the ELF and the
`libcudart`/`libcublas`/`libcublasLt` sonames it needs, which `sparsa.py` puts on `LD_LIBRARY_PATH`.

The archives are named as the previous releases were:
`sparsa-v<version>-windows-x64-cuda.zip` and `sparsa-v<version>-linux-x64-cuda.tar.gz`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = "strata-minimax.exe" if os.name == "nt" else "strata-minimax"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cuda_bin() -> Path:
    for base in (os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME"),
                 r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"):
        if not base:
            continue
        for candidate in (Path(base) / "bin" / "x64", Path(base) / "bin"):
            if candidate.is_dir():
                return candidate
    raise SystemExit("package: no CUDA bin folder (set CUDA_PATH)")


def windows_libs(engine: Path) -> list[Path]:
    """The DLLs `engine` imports that are *not* part of Windows: the CUDA runtime closure and the VC++ runtime.

    Anything else the import table names (`KERNEL32`, `SHELL32`, ...) is the operating system's, and the engine
    must not carry a private copy of it beside itself - that is the fastest way to break a release on another
    machine.  So a dependency is kept only when it resolves inside the CUDA toolkit, or when it is one of the
    VC++ runtime DLLs (which live in System32 on a machine with the redistributable installed).
    """
    dumpbin = shutil.which("dumpbin")
    if not dumpbin:
        for vs in sorted(Path(r"C:\Program Files\Microsoft Visual Studio").glob(
                "*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"), reverse=True):
            dumpbin = str(vs)
        if not dumpbin:
            raise SystemExit("package: dumpbin not found (run from a VS developer prompt)")
    cuda = cuda_bin().resolve()
    redist = ("msvcp", "vcruntime", "concrt", "ucrtbase")

    def deps(path: Path) -> list[str]:
        out = subprocess.run([dumpbin, "/DEPENDENTS", str(path)], capture_output=True, text=True,
                             errors="replace").stdout
        return [line.strip() for line in out.splitlines()
                if line.strip().lower().endswith(".dll") and not line.startswith("Dump")]

    found: dict[str, Path] = {}
    todo = deps(engine)
    while todo:
        name = todo.pop()
        low = name.lower()
        if low in found:
            continue
        resolved = None
        candidate = cuda / name
        if candidate.is_file():
            resolved = candidate
        elif low.startswith(redist):
            in_system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / name
            if in_system.is_file():
                resolved = in_system
        if resolved is None:
            continue        # a Windows DLL: the user's machine has it
        found[low] = resolved
        todo.extend(deps(resolved))
    if not any(p.suffix.lower() == ".dll" and p.name.lower().startswith("cublas") for p in found.values()):
        raise SystemExit("package: the import closure found no cublas (is the exe built with CUDA?)")
    return sorted(found.values(), key=lambda p: p.name)


def stage(files: list[Path], staging: Path) -> None:
    bindir = staging / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, bindir / f.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="release version, e.g. 0.1.2")
    ap.add_argument("--platform", choices=("windows", "linux"), default="windows" if os.name == "nt" else "linux")
    ap.add_argument("--engine", default=str(ROOT / "build" / ENGINE))
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--sums", action="store_true", help="only (re)write SHA256SUMS.txt over the archives in --out")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.sums:
        archives = sorted(list(out.glob("*.zip")) + list(out.glob("*.tar.gz")))
        if not archives:
            raise SystemExit(f"package: no archives in {out}")
        lines = [f"{sha256(p)}  {p.name}" for p in archives]
        (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0

    engine = Path(a.engine)
    if not engine.is_file():
        raise SystemExit(f"package: no engine at {engine}")
    libs = windows_libs(engine) if a.platform == "windows" else linux_libs(engine)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / f"sparsa-v{a.version}"
        stage([engine, *libs], staging)
        if a.platform == "windows":
            archive = out / f"sparsa-v{a.version}-windows-x64-cuda.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for f in sorted(staging.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(staging))
        else:
            archive = out / f"sparsa-v{a.version}-linux-x64-cuda.tar.gz"
            with tarfile.open(archive, "w:gz") as t:
                for f in sorted(staging.rglob("*")):
                    if f.is_file():
                        t.add(f, arcname=str(f.relative_to(staging)))
        size = archive.stat().st_size
        digest = sha256(archive)
    print(f"{archive}  {size} bytes  sha256 {digest}")
    return 0


def linux_libs(engine: Path) -> list[Path]:
    """The sonames `ldd` reports as not found or as libcuda/libcublas (the driver's libcuda stays the user's)."""
    out = subprocess.run(["ldd", str(engine)], capture_output=True, text=True, errors="replace").stdout
    keep = []
    for line in out.splitlines():
        line = line.strip()
        if "=>" not in line:
            continue
        name, rest = line.split("=>", 1)
        rest = rest.strip()
        if rest.startswith("not found"):
            raise SystemExit(f"package: {name.strip()} is not found on this build machine")
        path = Path(rest.split(" (")[0])
        if path.name.startswith(("libcublas", "libcudart")):
            keep.append(path)
    if not keep:
        raise SystemExit("package: ldd shows no CUDA runtime libraries")
    return sorted(set(keep), key=lambda p: p.name)


if __name__ == "__main__":
    sys.exit(main())
