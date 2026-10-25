#!/usr/bin/env python3
"""tools/minimax/pc_launch.py - the one way to start an engine (or a bench, or a g3 run) on the shared PC.

The PC runs one memory-hungry job at a time, and two out-of-memory crashes proved that "please take the lock and
check the RAM" in a brief is not enough.  This launcher makes it mechanical:

    python tools/minimax/pc_launch.py --owner dspark --est-ram-gb 86 -- \\
        build/strata-minimax.exe --model <full model> --serve ...

What it does, in order:

1. TAKES THE LOCK (`pc-engine.lock`, plain ASCII, no BOM) and refuses if another owner holds it with a live PID.
     - the file is `free` when unused, else `<owner> <pid> <iso-time>` followed by a `tree <pids...>` line
       recording the child process tree it started; a bare name with no PID (an older task's hand-written
       lock) counts as held. `--who` prints both, so a task can check before touching any process;
     - a PID that is no longer running is STALE: the launcher says so and takes over;
     - it is taken atomically: a sidecar `pc-engine.lock.lck` created with O_EXCL is the mutex, and the lock itself is
       written to a temp file and renamed over (so a reader never sees a half-written lock).
2. REFUSES TO START IF THE RAM WILL NOT FIT: effective available RAM (available physical minus modified/dirty
   file-cache pages, from PDH `\Memory\Modified Page List Bytes` on Windows or `/proc/meminfo` on Linux)
   minus the run's estimate must stay at or above `--min-free-gb` (default 10). The estimate is `--est-ram-gb`,
   or `estimate_ram_gb()` reading the engine's own flags (pinned expert budget, dense host copy, KV, prefill
   workspace, margin); the breakdown and both RAM numbers are printed.
3. WATCHES the child: every `--poll` seconds (default 2), if effective RAM falls below `--min-free-gb` it stops
   the child's whole process tree and exits 4 with the available, modified and effective readings that triggered it.
4. ALWAYS RELEASES: `try/finally` plus SIGINT/SIGTERM handlers put the lock back to `free` - on a clean exit, a child
   crash, Ctrl-C, or a watchdog kill.

Exit codes: 0..n = the child's own code, 2 = another owner holds the lock, 3 = not enough RAM, 4 = the watchdog
stopped the run, 5 = the lock could not be taken (the sidecar timed out), 130 = interrupted.

Stdout and stderr are the child's (the lock/RAM lines go to stderr, prefixed `[pc_launch]`), and stdin is inherited,
so an engine's serve protocol (GEN lines on stdin) works unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import errno
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_LOCK = Path(r"E:\Jon's Projects\AI\Strata Fork Minimax\pc-engine.lock")
DEFAULT_MIN_FREE_GB = 10.0
LOCK_FREE = b"free"

# ---- exit codes the callers (and the tests) rely on
EXIT_LOCK_HELD = 2
EXIT_NO_RAM = 3
EXIT_WATCHDOG = 4
EXIT_LOCK_TIMEOUT = 5

# Windows: still-running exit code of a process handle (GetExitCodeProcess -> STILL_ACTIVE)
_STILL_ACTIVE = 259


def log(msg: str) -> None:
    """Everything this tool says goes to stderr, so the child owns stdout."""
    sys.stderr.write("[pc_launch] " + msg + "\n")
    sys.stderr.flush()


# ============================================================================================== RAM (no PowerShell)
class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_gb() -> tuple[float, float]:
    """(total, available) physical RAM in GiB.  GlobalMemoryStatusEx on Windows, /proc/meminfo elsewhere."""
    if sys.platform == "win32":
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            raise OSError("GlobalMemoryStatusEx failed")
        gb = 1024.0 ** 3
        return st.ullTotalPhys / gb, st.ullAvailPhys / gb
    info = Path("/proc/meminfo").read_text(encoding="ascii", errors="replace")
    total = int(re.search(r"^MemTotal:\s+(\d+) kB", info, re.M).group(1)) * 1024
    avail = int(re.search(r"^MemAvailable:\s+(\d+) kB", info, re.M).group(1)) * 1024
    gb = 1024.0 ** 3
    return total / gb, avail / gb


# ---- modified (dirty) file-cache pages
class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double),
                    ("largeValue", ctypes.c_longlong), ("AnsiStringValue", ctypes.c_char_p),
                    ("WideStringValue", ctypes.c_wchar_p)]
    _anonymous_ = ("u",)
    _fields_ = [("CStatus", ctypes.c_ulong), ("u", _U)]


def modified_page_bytes() -> int:
    """Dirty file-cache pages: Windows `\\Memory\\Modified Page List Bytes` via PDH, Linux Meminfo Dirty+Writeback.

    Downloads leave dirty pages that MemoryAvailable still counts as available, so the launcher must take them out
    of the RAM it expects the engine to get. A counter that cannot be read is 0, never an invented number."""
    if sys.platform == "win32":
        pdh = ctypes.windll.pdh
        query = ctypes.c_void_p()
        counter = ctypes.c_void_p()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
            return 0
        try:
            add = getattr(pdh, "PdhAddEnglishCounterW", None) or pdh.PdhAddCounterW
            if add(query, "\\Memory\\Modified Page List Bytes", 0, ctypes.byref(counter)) != 0:
                return 0
            if pdh.PdhCollectQueryData(query) != 0:
                return 0
            value = _PDH_FMT_COUNTERVALUE()
            if pdh.PdhGetFormattedCounterValue(counter, 0x00000200, None, ctypes.byref(value)) != 0:
                return 0
            return int(value.doubleValue)
        finally:
            pdh.PdhCloseQuery(query)
    try:
        info = Path("/proc/meminfo").read_text(encoding="ascii", errors="replace")
        total = 0
        for key in ("Dirty", "Writeback"):
            m = re.search(rf"^{key}:\s+(\d+) kB", info, re.M)
            if m:
                total += int(m.group(1)) * 1024
        return total
    except OSError:
        return 0


def modified_page_gb() -> float:
    return modified_page_bytes() / (1024.0 ** 3)


def process_tree(root: int) -> tuple[int, ...]:
    """`root` and every descendant currently visible, parent before child. Used only for the lock record."""
    if root <= 0:
        return ()
    children: dict[int, list[int]] = {}
    if sys.platform == "win32":
        TH32CS_SNAPPROCESS = 0x00000002

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong),
                        ("th32ProcessID", ctypes.c_ulong), ("th32DefaultHeapID", ctypes.c_void_p),
                        ("th32ModuleID", ctypes.c_ulong), ("cntThreads", ctypes.c_ulong),
                        ("th32ParentProcessID", ctypes.c_ulong), ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", ctypes.c_ulong), ("szExeFile", ctypes.c_wchar * 260)]

        kernel32 = ctypes.windll.kernel32
        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == ctypes.c_void_p(-1).value:
            return (root,)
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                children.setdefault(int(entry.th32ParentProcessID), []).append(int(entry.th32ProcessID))
                ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snap)
    else:
        try:
            for path in Path("/proc").iterdir():
                if not path.name.isdigit():
                    continue
                try:
                    stat = (path / "stat").read_text(encoding="ascii", errors="replace")
                    ppid = int(stat.rsplit(")", 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
                children.setdefault(ppid, []).append(int(path.name))
        except OSError:
            return (root,)
    out, queue, seen = [], [root], set()
    while queue:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        queue.extend(sorted(children.get(pid, ())))
    return tuple(out)


class RamSource:
    """Where the RAM reading comes from: the machine, or (tests and dry runs only) a scripted sequence in GB.

    `auto` measures. `seq:20,20,3` returns 20, then 20, then 3 (and 3 for every later poll) without touching the
    machine - it is how the watchdog's decision is tested on a box whose free RAM this test cannot control.
    """

    def __init__(self, spec: str = "auto", modified_spec: str = "auto"):
        self.fixed = self._sequence(spec, "--ram-source")
        self.modified_fixed = self._sequence(modified_spec, "--modified-source")
        self.i = 0
        self.mi = 0

    @staticmethod
    def _sequence(spec: str, name: str) -> list[float]:
        if not spec or spec == "auto":
            return []
        if not spec.startswith("seq:"):
            raise ValueError(f"{name} wants 'auto' or 'seq:g1,g2,...', got {spec!r}")
        return [float(x) for x in spec[len("seq:"):].split(",") if x.strip()]

    def total(self) -> float:
        return memory_gb()[0]

    def available(self) -> float:
        if not self.fixed:
            return memory_gb()[1]
        v = self.fixed[min(self.i, len(self.fixed) - 1)]
        self.i += 1
        return v

    def modified(self) -> float:
        if not self.modified_fixed:
            return modified_page_gb()
        v = self.modified_fixed[min(self.mi, len(self.modified_fixed) - 1)]
        self.mi += 1
        return v


def pid_alive(pid: int) -> bool:
    """Whether `pid` is a running process.  A PID we may not query is treated as alive (it is not ours to break)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            # OpenProcess fails for a dead PID; error 87 (invalid parameter) means it is gone.  Anything else
            # (access denied) means it exists and we simply cannot look at it.
            return kernel32.GetLastError() != 87
        try:
            code = ctypes.c_ulong(0)
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ============================================================================================== the lock
class LockState:
    """What `pc-engine.lock` says: free, held by a live owner, held by a dead one, or a bare name (held)."""

    def __init__(self, owner: str = "free", pid: int | None = None, stamp: str = "", raw: str = "",
                 bare: bool = False, tree: tuple[int, ...] = ()):
        self.owner = owner
        self.pid = pid
        self.stamp = stamp
        self.raw = raw
        self.bare = bare
        self.tree = tree

    @property
    def free(self) -> bool:
        return self.owner == "free"

    @property
    def stale(self) -> bool:
        """A named owner whose PID is not running any more.  A bare name cannot be checked, so it is not stale."""
        return (not self.free) and (not self.bare) and self.pid is not None and not pid_alive(self.pid)

    def describe(self) -> str:
        if self.free:
            return "free"
        if self.bare:
            return f"held by '{self.owner}' (no PID on the line: an old-style lock, treated as held)"
        state = "stale PID" if self.stale else "live PID"
        return f"held by '{self.owner}' pid {self.pid} since {self.stamp} ({state})"

    def __str__(self) -> str:
        return self.describe()


def parse_lock(text: str) -> LockState:
    raw = text.replace("\ufeff", "").strip()
    if raw == "" or raw.lower().startswith("free"):
        return LockState(raw=raw)
    lines = raw.splitlines()
    parts = lines[0].split()
    owner = parts[0]
    tree: tuple[int, ...] = ()
    for line in lines[1:]:
        if line.startswith("tree "):
            tree = tuple(int(x) for x in line[5:].split() if x.isdigit())
    if len(parts) >= 2 and parts[1].isdigit():
        return LockState(owner=owner, pid=int(parts[1]), stamp=" ".join(parts[2:]), raw=raw, tree=tree)
    return LockState(owner=owner, raw=raw, bare=True, tree=tree)


def read_lock(path: Path) -> LockState:
    try:
        return parse_lock(path.read_text(encoding="ascii", errors="replace"))
    except FileNotFoundError:
        return LockState(raw="")


def _write_atomic(path: Path, data: bytes) -> None:
    """`data` to `path` through a temp file and a rename: a reader never sees a half-written lock."""
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_lock_free(path: Path) -> None:
    _write_atomic(path, LOCK_FREE)


def write_lock_owner(path: Path, owner: str, pid: int, stamp: str | None = None,
                     tree: Iterable[int] = ()) -> None:
    stamp = stamp or datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [f"{owner} {pid} {stamp}"]
    pids = [int(x) for x in tree if int(x) > 0]
    if pids:
        lines.append("tree " + " ".join(str(x) for x in pids))
    _write_atomic(path, ("\n".join(lines) + "\n").encode("ascii"))


def write_lock_tree(path: Path, owner: str, owner_pid: int, tree: Iterable[int]) -> bool:
    """Update the held lock's child-PID record, if it is still this launcher's lock."""
    with lock_guard(path, 10.0):
        state = read_lock(path)
        if state.free or state.owner != owner or state.pid != owner_pid:
            return False
        write_lock_owner(path, owner, owner_pid, state.stamp, tree)
        return True


class LockBusy(RuntimeError):
    def __init__(self, state: LockState):
        super().__init__(state.describe())
        self.state = state


class lock_guard:
    """The sidecar mutex around the read-modify-write of the lock (O_EXCL, stale sidecars recovered)."""

    def __init__(self, lock: Path, timeout: float = 10.0, stale_after: float = 60.0):
        self.path = lock.with_name(lock.name + ".lck")
        self.timeout = timeout
        self.stale_after = stale_after
        self.fd = -1

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(self.fd, f"{os.getpid()} {time.time()}".encode("ascii"))
                return self
            except FileExistsError:
                # a crashed holder leaves the sidecar behind: age it out instead of wedging the PC
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_after:
                    log(f"removing a stale lock mutex {self.path.name} ({age:.0f}s old)")
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"the lock mutex {self.path} stayed busy for {self.timeout:.0f}s")
                time.sleep(0.02)

    def __exit__(self, *exc):
        if self.fd >= 0:
            try:
                os.close(self.fd)
            finally:
                self.fd = -1
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        return False


def acquire(lock: Path, owner: str, timeout: float = 10.0) -> LockState:
    """Take the lock for `owner` (this process's PID) and return the state that was there.  Raises LockBusy."""
    with lock_guard(lock, timeout):
        state = read_lock(lock)
        if not state.free and not state.stale:
            raise LockBusy(state)
        if state.free:
            log(f"lock is free, taking it for '{owner}'")
        else:
            log(f"lock was {state.describe()} -> taking it over for '{owner}'")
        write_lock_owner(lock, owner, os.getpid())
    return state


def release(lock: Path, owner: str) -> None:
    """Put the lock back to `free` - unless someone else has taken it meanwhile (then say so, do not clobber)."""
    with lock_guard(lock, 10.0):
        state = read_lock(lock)
        if not state.free and state.owner != owner:
            log(f"not releasing: the lock now belongs to '{state.owner}' (pid {state.pid})")
            return
        write_lock_free(lock)
    log("lock released (free)")


# ============================================================================================== the RAM estimate
# The engine's own shapes: MiniMax-M3.1-Flash at IQ2_XS-D3 is what these defaults describe (57 MoE layers, an
# IQ2_XS expert of 14.6 MB, 135 KB of KV+indexer keys per token).
MOE_LAYERS = 57
EXPERT_GB = 14.6 / 1024.0
KV_GB_PER_TOKEN = 135.0 / 1024.0 / 1024.0


def _flag(argv: Sequence[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def estimate_ram_gb(argv: Sequence[str], total_gb: float) -> tuple[float, list[str]]:
    """A run's host-RAM need from the engine's flags, with the line that produced each part.

    The parts are the ones that actually hold memory on this box: the pinned expert tier (`--memory-limit-mib`, or
    `--minimax-ram-experts-per-layer x 57 x 14.6 MB`), a host copy of the dense weights when they are not on a card
    (`--minimax-dense-layers-gpu 0`), the KV cache when it is not in VRAM (`--max-context x 135 KB`), the batched
    prompt workspace, and a fixed margin for the process and the driver.
    """
    parts: list[str] = []
    limit = _flag(argv, "--memory-limit-mib")
    prelayer = _flag(argv, "--minimax-ram-experts-per-layer")
    dense_gpu = _flag(argv, "--minimax-dense-layers-gpu")
    ctx = _flag(argv, "--max-context") or _flag(argv, "--ctx")
    kv_space = _flag(argv, "--minimax-kv") or _flag(argv, "--kv-space") or "auto"
    batch = _flag(argv, "--minimax-prefill-batch") or _flag(argv, "--minimax-batch")
    headroom = _flag(argv, "--minimax-ram-headroom-mib")

    pinned = 0.0
    if limit:
        pinned = float(limit) / 1024.0
        parts.append(f"pinned expert tier {pinned:.1f} GB (--memory-limit-mib {limit})")
    elif prelayer and int(prelayer) > 0:
        pinned = int(prelayer) * MOE_LAYERS * EXPERT_GB
        parts.append(f"pinned expert tier {pinned:.1f} GB (--minimax-ram-experts-per-layer {prelayer} x "
                     f"{MOE_LAYERS} layers x 14.6 MB)")
    else:
        pinned = 0.35 * total_gb
        parts.append(f"pinned expert tier {pinned:.1f} GB (no budget given: 35% of physical RAM)")
    dense = 2.0 if dense_gpu is None or dense_gpu != "0" else 12.0
    parts.append(f"dense host copy {dense:.1f} GB (--minimax-dense-layers-gpu {dense_gpu or 'auto'})")
    kv = 0.0
    if ctx and kv_space in ("host", "ram"):
        kv = int(ctx) * KV_GB_PER_TOKEN
        parts.append(f"KV {kv:.2f} GB ({ctx} tokens x 135 KB in {kv_space})")
    else:
        parts.append("KV 0.1 GB (on the cards)")
        kv = 0.1
    ws = 0.7 if batch and int(batch) >= 1024 else 0.1
    parts.append(f"prefill workspace {ws:.1f} GB (--minimax-prefill-batch {batch or '0'})")
    margin = 1.5 + (float(headroom) / 1024.0 if headroom else 0.0)
    parts.append(f"margin {margin:.1f} GB (process + driver)")
    return pinned + dense + kv + ws + margin, parts


# ============================================================================================== the child
def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, text=True)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _spawn(argv: Sequence[str]) -> subprocess.Popen:
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(list(argv), **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Exit codes: the child's own, 2 lock held, 3 not enough RAM, 4 watchdog, "
                                        "5 lock timeout, 130 interrupted.")
    ap.add_argument("--owner", help="who takes the lock ('dspark', 'maya', ...)")
    ap.add_argument("--est-ram-gb", type=float, default=None,
                    help="the run's host-RAM need; without it the engine flags are read (see estimate_ram_gb)")
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
                    help="available RAM the run must leave behind (default 10)")
    ap.add_argument("--lock", default=os.environ.get("STRATA_PC_LOCK", str(DEFAULT_LOCK)),
                    help="the lock file (default %(default)s, STRATA_PC_LOCK overrides)")
    ap.add_argument("--lock-timeout", type=float, default=10.0, help="how long to wait for the lock mutex")
    ap.add_argument("--poll", type=float, default=2.0, help="the watchdog's interval in seconds (default 2)")
    ap.add_argument("--no-watchdog", action="store_true", help="do not stop the run on low RAM (not recommended)")
    ap.add_argument("--dry-run", action="store_true", help="take the lock, check the RAM, print, release, do nothing")
    ap.add_argument("--ram-source", default="auto",
                    help="where the RAM reading comes from: 'auto' measures; 'seq:g1,g2,...' is a scripted sequence "
                         "in GB, for the tests and dry runs (it never touches the machine)")
    ap.add_argument("--modified-source", default="auto",
                    help="dirty-page reading: 'auto' (PDH / /proc/meminfo) or 'seq:g1,g2,...' for tests")
    ap.add_argument("--who", action="store_true",
                    help="print the lock owner and the process tree it recorded, then exit")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- the command to run")
    a = ap.parse_args(argv)
    lock = Path(a.lock)

    if a.who:
        state = read_lock(lock)
        print(f"owner={state.owner} pid={state.pid or 0} stamp={state.stamp or '-'} "
              f"state={'free' if state.free else 'held'}")
        if not state.free:
            print("tree=" + " ".join(str(x) for x in state.tree))
            for pid in state.tree:
                print(f"  {pid} alive={'yes' if pid_alive(pid) else 'no'}")
        return 0

    if not a.owner:
        ap.error("--owner is required")
    try:
        ram = RamSource(a.ram_source, a.modified_source)
    except ValueError as e:
        ap.error(str(e))

    cmd = [c for c in a.cmd if c != "--"]
    if not cmd:
        ap.error("a command is required after --")

    try:
        was = acquire(lock, a.owner, a.lock_timeout)
    except LockBusy as e:
        log(f"REFUSING: the lock is {e.state.describe()}")
        log("another task is using the PC; take your turn when it writes 'free'")
        return EXIT_LOCK_HELD
    except TimeoutError as e:
        log(f"REFUSING: {e}")
        return EXIT_LOCK_TIMEOUT

    child: subprocess.Popen | None = None
    stop_signals = {"received": False}

    def on_signal(signum, _frame):
        stop_signals["received"] = True
        if child is not None and child.poll() is None:
            log(f"signal {signum}: stopping the child tree")
            _kill_tree(child.pid)
        raise SystemExit(130)

    old_handlers = {}
    sigs = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        sigs.append(signal.SIGBREAK)   # Windows' Ctrl-Break / CTRL_BREAK_EVENT
    for sig in sigs:
        try:
            old_handlers[sig] = signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass

    try:
        total_gb, avail_gb, mod_gb = ram.total(), ram.available(), ram.modified()
        effective_gb = max(0.0, avail_gb - mod_gb)
        if ram.fixed:
            log(f"RAM reading is scripted (--ram-source {a.ram_source})")
        if ram.modified_fixed:
            log(f"dirty-page reading is scripted (--modified-source {a.modified_source})")
        if a.est_ram_gb is not None:
            est, parts = a.est_ram_gb, [f"estimate given on the command line ({a.est_ram_gb:.1f} GB)"]
        else:
            est, parts = estimate_ram_gb(cmd, total_gb)
        for p in parts:
            log("  " + p)
        log(f"RAM: {avail_gb:.1f} GB available of {total_gb:.1f} GB, {mod_gb:.1f} GB in modified (dirty) pages "
            f"-> {effective_gb:.1f} GB effective; this run needs about {est:.1f} GB")
        left = effective_gb - est
        if left < a.min_free_gb:
            log(f"REFUSING: {left:.1f} GB would be left, under the {a.min_free_gb:.0f} GB floor "
                f"(effective {effective_gb:.1f} GB = available {avail_gb:.1f} - modified {mod_gb:.1f}, "
                f"estimate {est:.1f} GB)")
            return EXIT_NO_RAM
        log(f"{left:.1f} GB would be left free (floor {a.min_free_gb:.0f} GB)")

        if a.dry_run:
            log("dry run: not starting the command")
            return 0

        log("starting: " + " ".join(cmd))
        t0 = time.monotonic()
        child = _spawn(cmd)
        next_tree = 0.0
        while True:
            if time.monotonic() >= next_tree:
                try:
                    write_lock_tree(lock, a.owner, os.getpid(), process_tree(child.pid))
                except OSError as e:
                    log(f"could not record the child tree in the lock: {e}")
                next_tree = time.monotonic() + 10.0
            try:
                rc = child.wait(timeout=a.poll)
                break
            except subprocess.TimeoutExpired:
                if a.no_watchdog or stop_signals["received"]:
                    continue
                now_gb, dirty_gb = ram.available(), ram.modified()
                effective_now = max(0.0, now_gb - dirty_gb)
                if effective_now < a.min_free_gb:
                    log(f"WATCHDOG: {effective_now:.1f} GB effective is under the {a.min_free_gb:.0f} GB floor "
                        f"(available {now_gb:.1f}, modified {dirty_gb:.1f}); stopping pid {child.pid} and its children")
                    _kill_tree(child.pid)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    return EXIT_WATCHDOG
        log(f"child exited {rc} after {time.monotonic() - t0:.0f}s")
        return rc
    finally:
        if child is not None and child.poll() is None:
            log("stopping the child tree (the launcher is leaving)")
            _kill_tree(child.pid)
        release(lock, a.owner)
        for sig, h in old_handlers.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError):
                pass


if __name__ == "__main__":
    sys.exit(main())
