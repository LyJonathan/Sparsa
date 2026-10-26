"""tools/minimax/pc_dump.py - what a hung engine is waiting on, without WinDbg.

Three things, because a hang needs all of them:

  `dump`    write a minidump (dbghelp `MiniDumpWriteDump`, thread stacks + TEB + module list, NOT full memory: the
            engine's working set is tens of GB) with the 64-bit process handle passed correctly.
  `stacks`  walk every thread's stack *live* with dbghelp `StackWalk64` and resolve module!function+0xoffset from the
            PDB beside the executable. Every thread is suspended for the walk and resumed after it, so the frames are
            consistent.
  `read`    open a saved minidump offline and print its module list and, for every thread, the instruction pointer and
            the module-mapped return addresses scanned off that thread's stack. No symbols (a dump carries none), but
            `ntdll+0x...` / `kernel32+0x...` / `strata-minimax+0x...` is enough to see which wait it is.

    python tools/minimax/pc_dump.py stacks --pid 1234
    python tools/minimax/pc_dump.py dump --pid 1234 --out hung.dmp
    python tools/minimax/pc_dump.py read hung.dmp
    python tools/minimax/pc_dump.py sample --pid 1234          # cheap: thread states + wait reasons, no suspend

The test hook is `dummy`, which starts a process with four threads in four known waits and runs the whole path on it:

    python tools/minimax/pc_dump.py selftest
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

IS_WIN = os.name == "nt"
if not IS_WIN:  # the engine only ships on Windows; keep --help working elsewhere
    print("pc_dump: Windows only", file=sys.stderr)

k32 = ctypes.WinDLL("kernel32", use_last_error=True) if IS_WIN else None
dbg = ctypes.WinDLL("dbghelp", use_last_error=True) if IS_WIN else None

# ---------------------------------------------------------------- types
TH32CS_SNAPTHREAD = 0x4
THREAD_SUSPEND_RESUME, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION = 0x2, 0x8, 0x40
PROCESS_ALL_ACCESS = 0x1F0FFF
IMAGE_FILE_MACHINE_AMD64 = 0x8664
MINIDUMP_NORMAL, MINIDUMP_WITH_THREAD_INFO, MINIDUMP_WITH_INDIRECT = 0, 0x1000, 0x40
ADDRESS_MODE_FLAT = 3
SYMOPT_UNDNAME, SYMOPT_DEFERRED_LOADS, SYMOPT_LOAD_LINES = 0x2, 0x4, 0x10
MAX_SYM_NAME = 2000


class ADDRESS64(ctypes.Structure):
    _fields_ = [("Offset", ctypes.c_uint64), ("Segment", ctypes.c_uint16), ("Mode", ctypes.c_uint32)]


class STACKFRAME64(ctypes.Structure):
    _fields_ = [("AddrPC", ADDRESS64), ("AddrReturn", ADDRESS64), ("AddrFrame", ADDRESS64),
                ("AddrStack", ADDRESS64), ("AddrBStore", ADDRESS64), ("FuncTableEntry", ctypes.c_void_p),
                ("Params", ctypes.c_uint64 * 4), ("Far", wt.BOOL), ("Virtual", wt.BOOL),
                ("Reserved", ctypes.c_uint64 * 3)]


class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.ULONG), ("TypeIndex", wt.ULONG), ("Reserved", ctypes.c_uint64 * 2),
                ("Index", wt.ULONG), ("Size", wt.ULONG), ("ModBase", ctypes.c_uint64), ("Flags", wt.ULONG),
                ("Value", ctypes.c_uint64), ("Address", ctypes.c_uint64), ("Register", wt.ULONG),
                ("Scope", wt.ULONG), ("Tag", wt.ULONG), ("NameLen", wt.ULONG), ("MaxNameLen", wt.ULONG),
                ("Name", wt.WCHAR * (MAX_SYM_NAME + 1))]


class IMAGEHLP_MODULEW64(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.ULONG), ("BaseOfImage", ctypes.c_uint64), ("ImageSize", wt.ULONG),
                ("TimeDateStamp", wt.ULONG), ("CheckSum", wt.ULONG), ("NumSyms", wt.ULONG),
                ("SymType", wt.ULONG), ("ModuleName", wt.WCHAR * 32), ("ImageName", wt.WCHAR * 256),
                ("LoadedImageName", wt.WCHAR * 256), ("LoadedPdbName", wt.WCHAR * 256),
                ("CvSig", wt.ULONG), ("ElfSig", wt.ULONG), ("CvSig2", wt.ULONG), ("PdbSig", wt.ULONG),
                ("PdbSig70", ctypes.c_ubyte * 16), ("PdbAge", wt.ULONG), ("PdbUnmatched", wt.BOOL),
                ("DbgUnmatched", wt.BOOL), ("LineNumbers", wt.BOOL), ("GlobalSymbols", wt.BOOL),
                ("TypeInfo", wt.BOOL), ("SourceIndexed", wt.BOOL), ("Publics", wt.BOOL),
                ("MachineType", wt.ULONG), ("Reserved0", wt.ULONG), ("Reserved1", wt.ULONG),
                ("Reserved2", wt.ULONG), ("Reserved3", wt.ULONG), ("Reserved4", wt.ULONG),
                ("Reserved5", wt.ULONG), ("Reserved6", wt.ULONG), ("Reserved7", wt.ULONG),
                ("Reserved8", wt.ULONG), ("Reserved9", wt.ULONG)]


class THREADENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ThreadID", wt.DWORD),
                ("th32OwnerProcessID", wt.DWORD), ("tpBasePri", wt.LONG), ("tpDeltaPri", wt.LONG),
                ("dwFlags", wt.DWORD)]


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("th32ModuleID", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("GlblcntUsage", wt.DWORD), ("ProccntUsage", wt.DWORD), ("modBaseAddr", ctypes.c_void_p),
                ("modBaseSize", wt.DWORD), ("hModule", ctypes.c_void_p),
                ("szModule", wt.WCHAR * 256), ("szExePath", wt.WCHAR * 260)]


class CONTEXT64(ctypes.Structure):
    _fields_ = [("P1Home", ctypes.c_uint64 * 6), ("ContextFlags", wt.DWORD), ("MxCsr", wt.DWORD),
                ("SegCs", wt.WORD), ("SegDs", wt.WORD), ("SegEs", wt.WORD), ("SegFs", wt.WORD),
                ("SegGs", wt.WORD), ("SegSs", wt.WORD), ("EFlags", wt.DWORD),
                ("Dr", ctypes.c_uint64 * 6), ("Rax", ctypes.c_uint64), ("Rcx", ctypes.c_uint64),
                ("Rdx", ctypes.c_uint64), ("Rbx", ctypes.c_uint64), ("Rsp", ctypes.c_uint64),
                ("Rbp", ctypes.c_uint64), ("Rsi", ctypes.c_uint64), ("Rdi", ctypes.c_uint64),
                ("R8", ctypes.c_uint64), ("R9", ctypes.c_uint64), ("R10", ctypes.c_uint64),
                ("R11", ctypes.c_uint64), ("R12", ctypes.c_uint64), ("R13", ctypes.c_uint64),
                ("R14", ctypes.c_uint64), ("R15", ctypes.c_uint64), ("Rip", ctypes.c_uint64)]


def _bind():
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    k32.Thread32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(THREADENTRY32)]
    k32.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(THREADENTRY32)]
    k32.OpenThread.restype = ctypes.c_void_p
    k32.OpenThread.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k32.SuspendThread.argtypes = [ctypes.c_void_p]
    k32.ResumeThread.argtypes = [ctypes.c_void_p]
    k32.GetThreadContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.ReadProcessMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_size_t)]
    dbg.MiniDumpWriteDump.restype = wt.BOOL
    dbg.MiniDumpWriteDump.argtypes = [ctypes.c_void_p, wt.DWORD, wt.HANDLE, wt.DWORD,
                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    dbg.SymInitializeW.argtypes = [ctypes.c_void_p, wt.LPCWSTR, wt.BOOL]
    dbg.SymSetOptions.argtypes = [wt.DWORD]
    dbg.SymFromAddrW.restype = wt.BOOL
    dbg.SymFromAddrW.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                                 ctypes.POINTER(SYMBOL_INFOW)]
    dbg.SymGetModuleInfoW64.restype = wt.BOOL
    dbg.SymGetModuleInfoW64.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(IMAGEHLP_MODULEW64)]
    dbg.StackWalk64.restype = wt.BOOL
    dbg.StackWalk64.argtypes = [wt.DWORD, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(STACKFRAME64),
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p]
    # 64-bit addresses: without these the callbacks' second argument is truncated and raises OverflowError
    dbg.SymFunctionTableAccess64.restype = ctypes.c_void_p
    dbg.SymFunctionTableAccess64.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    dbg.SymGetModuleBase64.restype = ctypes.c_uint64
    dbg.SymGetModuleBase64.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    dbg.SymCleanup.argtypes = [ctypes.c_void_p]
    dbg.SymLoadModuleExW.restype = ctypes.c_uint64
    dbg.SymLoadModuleExW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.LPCWSTR, wt.LPCWSTR, ctypes.c_uint64,
                                     wt.DWORD, ctypes.c_void_p, wt.DWORD]
    k32.Module32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MODULEENTRY32W)]
    k32.Module32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MODULEENTRY32W)]


if IS_WIN:
    _bind()

Threads = tuple[int, ...]


def modules_of(pid: int) -> list[dict]:
    """The target's loaded modules (base, size, name, path): what SymLoadModuleExW needs, and the fallback map."""
    TH32CS_SNAPMODULE, TH32CS_SNAPMODULE32 = 0x8, 0x10
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if not snap or snap == ctypes.c_void_p(-1).value:
        return []
    try:
        me = MODULEENTRY32W()
        me.dwSize = ctypes.sizeof(MODULEENTRY32W)
        out = []
        ok = k32.Module32FirstW(snap, ctypes.byref(me))
        while ok:
            out.append({"base": me.modBaseAddr or 0, "size": me.modBaseSize,
                        "name": str(me.szModule), "path": str(me.szExePath)})
            ok = k32.Module32NextW(snap, ctypes.byref(me))
        return out
    finally:
        k32.CloseHandle(snap)


_EXPORTS: dict[str, list[tuple[int, str]]] = {}


def export_map(path: str) -> list[tuple[int, str]]:
    """A PE file's exported functions as (rva, name), which names a system-DLL frame without a symbol server."""
    if path in _EXPORTS:
        return _EXPORTS[path]
    out: list[tuple[int, str]] = []
    try:
        data = Path(path).read_bytes()
        u16 = lambda o: struct.unpack_from("<H", data, o)[0]
        u32 = lambda o: struct.unpack_from("<I", data, o)[0]
        pe = u32(0x3C)
        if data[pe:pe + 4] == b"PE\0\0":
            coff = pe + 4
            nsec, opt_size = u16(coff + 2), u16(coff + 16)
            opt = coff + 20
            plus = u16(opt) == 0x20B
            dd = opt + (112 if plus else 96)
            exp_rva = u32(dd)
            secs = []
            so = opt + opt_size
            for i in range(nsec):
                b = so + i * 40
                secs.append((u32(b + 12), u32(b + 8), u32(b + 20), u32(b + 16)))

            def off(rva: int) -> int | None:
                for va, vsize, rawptr, rawsize in secs:
                    if va <= rva < va + max(vsize, rawsize):
                        return rawptr + (rva - va)
                return None

            if exp_rva:
                eo = off(exp_rva)
                if eo is not None:
                    nnames = u32(eo + 24)
                    funcs_rva, names_rva, ords_rva = u32(eo + 28), u32(eo + 32), u32(eo + 36)
                    no, oo = off(names_rva), off(ords_rva)
                    for i in range(nnames):
                        if no is None or oo is None:
                            break
                        nrva = u32(no + i * 4)
                        ord_i = u16(oo + i * 2)
                        fo = off(funcs_rva + ord_i * 4)
                        so_ = off(nrva)
                        if fo is None or so_ is None:
                            break
                        end = data.index(b"\0", so_)
                        out.append((u32(fo), data[so_:end].decode("ascii", "replace")))
    except (OSError, ValueError, struct.error, IndexError):
        out = []
    out.sort()
    _EXPORTS[path] = out
    return out


def export_name(mod: dict, addr: int) -> str:
    """`ntdll!NtWaitForSingleObject+0x14` style, from the module's own export table."""
    rva = addr - mod["base"]
    exps = export_map(mod["path"]) if mod.get("path") else []
    best = None
    for frva, name in exps:
        if frva <= rva:
            best = (frva, name)
        else:
            break
    if best:
        return f"{mod['name'].rsplit('.', 1)[0]}!{best[1]}+0x{rva - best[0]:x}"
    return f"{mod['name']}+0x{rva:x}"


def thread_ids(pid: int) -> list[int]:
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if not snap or snap == ctypes.c_void_p(-1).value:
        return []
    try:
        te = THREADENTRY32()
        te.dwSize = ctypes.sizeof(THREADENTRY32)
        out = []
        ok = k32.Thread32First(snap, ctypes.byref(te))
        while ok:
            if te.th32OwnerProcessID == pid:
                out.append(te.th32ThreadID)
            ok = k32.Thread32Next(snap, ctypes.byref(te))
        return out
    finally:
        k32.CloseHandle(snap)


# ---------------------------------------------------------------- minidump
def write_dump(pid: int, path: Path, with_full_memory: bool = False) -> tuple[bool, str]:
    """A minidump with every thread's stack (type 0 + thread info), never the engine's full working set."""
    h = k32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not h:
        return False, f"OpenProcess({pid}) failed: {ctypes.get_last_error()}"
    try:
        flags = MINIDUMP_NORMAL | MINIDUMP_WITH_THREAD_INFO
        if with_full_memory:
            flags |= 0x2  # MiniDumpWithFullMemory - only for a small process
        path.parent.mkdir(parents=True, exist_ok=True)
        import msvcrt
        with open(path, "wb") as f:
            os_handle = wt.HANDLE(msvcrt.get_osfhandle(f.fileno()))   # a CRT fd is not a win32 HANDLE
            ok = dbg.MiniDumpWriteDump(h, pid, os_handle, flags, None, None, None)
        if not ok:
            return False, f"MiniDumpWriteDump failed: {ctypes.get_last_error()}"
        return True, f"{path} ({path.stat().st_size / 1024:.0f} KiB)"
    finally:
        k32.CloseHandle(h)


MODULE_STREAMS = {4: "ModuleList", 3: "ThreadList", 5: "MemoryList", 7: "SystemInfo", 9: "Memory64List"}


def read_dump(path: Path) -> dict:
    """Parse the parts of a minidump this needs: modules, threads (context + stack range), memory."""
    data = path.read_bytes()
    if data[:4] != b"MDMP":
        raise SystemExit(f"{path}: not a minidump")
    (_sig, _ver, nstreams, dir_rva) = struct.unpack_from("<IIII", data, 0)
    dirs = {}
    for i in range(nstreams):
        stype, size, rva = struct.unpack_from("<III", data, dir_rva + i * 12)
        if size:                     # `size` is a union with the `rva` for some streams; both are 4 bytes here
            dirs[stype] = (size, rva)

    modules = []
    if 4 in dirs:
        size, rva = dirs[4]
        (nmod,) = struct.unpack_from("<I", data, rva)
        off = rva + 4
        for _ in range(nmod):
            base, size_img, _cs, _tds = struct.unpack_from("<QIII", data, off)
            name_rva, = struct.unpack_from("<I", data, off + 20)
            (nlen,) = struct.unpack_from("<I", data, name_rva)
            raw = data[name_rva + 4:name_rva + 4 + nlen]
            full = raw.decode("utf-16-le", "replace")
            modules.append({"base": base, "size": size_img, "name": os.path.basename(full), "path": full})
            off += 108                  # sizeof(MINIDUMP_MODULE)

    threads = []
    if 3 in dirs:
        size, rva = dirs[3]
        (nthr,) = struct.unpack_from("<I", data, rva)
        off = rva + 4
        for _ in range(nthr):
            tid, _susp, _pri_c, _pri, _teb = struct.unpack_from("<IIIIQ", data, off)
            stack_start, stack_size, mem_rva = struct.unpack_from("<QII", data, off + 24)
            ctx_size, ctx_rva = struct.unpack_from("<II", data, off + 40)   # ThreadContext follows the Stack
            rip = rsp = rbp = 0
            if ctx_size >= 0x100 and ctx_rva + 0x100 <= len(data):
                rsp, rbp = struct.unpack_from("<QQ", data, ctx_rva + 0x98)
                (rip,) = struct.unpack_from("<Q", data, ctx_rva + 0xF8)
            threads.append({"tid": tid, "rip": rip, "rsp": rsp, "rbp": rbp,
                            "stack_start": stack_start, "stack_size": stack_size, "stack_rva": mem_rva})
            off += 48                   # sizeof(MINIDUMP_THREAD)

    mem = {}
    if 5 in dirs:
        size, rva = dirs[5]
        (nr,) = struct.unpack_from("<I", data, rva)
        for i in range(nr):
            start, msize, mrva = struct.unpack_from("<QII", data, rva + 4 + i * 16)
            mem[start] = (msize, mrva)
    return {"modules": modules, "threads": threads, "memory": mem, "size": len(data)}


def _mod_for(modules, addr: int) -> str:
    for m in modules:
        if m["base"] <= addr < m["base"] + m["size"]:
            return f"{m['name']}+0x{addr - m['base']:x}"
    return f"0x{addr:x}"


def print_dump(path: Path) -> None:
    d = read_dump(path)
    print(f"{path}: {d['size'] / 1024:.0f} KiB, {len(d['modules'])} modules, {len(d['threads'])} threads")
    print("modules: " + ", ".join(f"{m['name']}@{m['base']:x}" for m in d["modules"][:12])
          + (" ..." if len(d["modules"]) > 12 else ""))

    def name_of(addr: int) -> str:
        hit = next((m for m in d["modules"] if m["base"] <= addr < m["base"] + m["size"]), None)
        if not hit:
            return f"0x{addr:x}"
        return export_name(hit, addr) if hit.get("path") else f"{hit['name']}+0x{addr - hit['base']:x}"

    for t in d["threads"]:
        print(f"\nthread {t['tid']}: rip {name_of(t['rip'])}  rsp {t['rsp']:x}  "
              f"stack {t['stack_size'] / 1024:.0f} KiB")
        if t["stack_rva"] == 0 or t["stack_size"] == 0:
            print("  (no stack memory in this dump)")
            continue
        stack = path.read_bytes()[t["stack_rva"]:t["stack_rva"] + t["stack_size"]]
        hits = []
        for i in range(0, min(len(stack) - 8, 8 * 1024), 8):     # scan the top of the stack for return addresses
            (v,) = struct.unpack_from("<Q", stack, i)
            hit = next((m for m in d["modules"] if m["base"] + 0x1000 <= v < m["base"] + m["size"]), None)
            if hit:
                hits.append(name_of(v))
            if len(hits) >= 8:
                break
        print("  return addresses: " + (" -> ".join(hits) if hits else "(none found in the top 8 KiB)"))


# ---------------------------------------------------------------- live stacks
_READPROC = ctypes.WINFUNCTYPE(wt.BOOL, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, wt.DWORD,
                               ctypes.POINTER(wt.DWORD))
_FUNC_TABLE = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64)
_MOD_BASE = ctypes.WINFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)
_CALLBACKS: list = []          # dbghelp keeps these trampolines for the walk; never let Python free them


def live_stacks(pid: int, exe_dir: str | None = None, max_frames: int = 24) -> list[str]:
    """Every thread's stack with module!function+offset, from the PDB beside the executable."""
    h = k32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not h:
        return [f"OpenProcess({pid}) failed: {ctypes.get_last_error()}"]
    out = []
    try:
        search = exe_dir or ""
        if not dbg.SymInitializeW(h, search or None, False):
            return [f"SymInitialize failed: {ctypes.get_last_error()}"]
        dbg.SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES)
        mods = modules_of(pid)
        for m in mods:
            # dbghelp does not enumerate another process's modules by itself: load each one, which is also what makes
            # the PDB beside the engine resolve its frames
            dbg.SymLoadModuleExW(h, None, m["path"] or None, m["name"] or None, m["base"], m["size"], None, 0)

        def _read(hp, addr, buf, n, read):
            got = ctypes.c_size_t(0)
            ok = k32.ReadProcessMemory(hp, ctypes.c_void_p(addr), buf, n, ctypes.byref(got))
            if read:
                read[0] = got.value
            return ok

        def _fta(hp, addr):
            return dbg.SymFunctionTableAccess64(hp, addr)

        def _base(hp, addr):
            return dbg.SymGetModuleBase64(hp, addr)

        read_cb, fta_cb, base_cb = _READPROC(_read), _FUNC_TABLE(_fta), _MOD_BASE(_base)
        # keep the trampolines alive for the process lifetime: dbghelp holds their pointers, and freeing them while the
        # interpreter shuts down is a use-after-free (0xC0000005 on exit)
        _CALLBACKS.extend([read_cb, fta_cb, base_cb])
        dbg.StackWalk64.restype = wt.BOOL

        tids = thread_ids(pid)
        handles = []
        for tid in tids:
            th = k32.OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, False, tid)
            if th:
                k32.SuspendThread(th)
                handles.append((tid, th))
        try:
            for tid, th in handles:
                ctx = CONTEXT64()
                ctx.ContextFlags = 0x00100003        # CONTEXT_AMD64 | CONTROL | INTEGER
                if not k32.GetThreadContext(th, ctypes.byref(ctx)):
                    out.append(f"thread {tid}: GetThreadContext failed ({ctypes.get_last_error()})")
                    continue
                frame = STACKFRAME64()
                frame.AddrPC.Offset, frame.AddrPC.Mode = ctx.Rip, ADDRESS_MODE_FLAT
                frame.AddrFrame.Offset, frame.AddrFrame.Mode = ctx.Rbp, ADDRESS_MODE_FLAT
                frame.AddrStack.Offset, frame.AddrStack.Mode = ctx.Rsp, ADDRESS_MODE_FLAT
                lines = [f"thread {tid} (pc {ctx.Rip:x})"]
                for _ in range(max_frames):
                    if not dbg.StackWalk64(IMAGE_FILE_MACHINE_AMD64, h, ctypes.c_void_p(th),
                                           ctypes.byref(frame), ctypes.byref(ctx),
                                           read_cb, fta_cb, base_cb, None):
                        break
                    addr = frame.AddrPC.Offset
                    if not addr:
                        break
                    sym = SYMBOL_INFOW()
                    sym.SizeOfStruct = ctypes.sizeof(SYMBOL_INFOW)
                    sym.MaxNameLen = MAX_SYM_NAME
                    disp = ctypes.c_uint64(0)
                    if dbg.SymFromAddrW(h, addr, ctypes.byref(disp), ctypes.byref(sym)):
                        lines.append(f"    {sym.Name.decode('utf-16-le', 'replace')}+0x{disp.value:x}")
                    else:
                        hit = next((m for m in mods if m["base"] <= addr < m["base"] + m["size"]), None)
                        if hit and hit.get("path"):
                            lines.append("    " + export_name(hit, addr))
                        else:
                            mi = IMAGEHLP_MODULEW64()
                            mi.SizeOfStruct = ctypes.sizeof(IMAGEHLP_MODULEW64)
                            if dbg.SymGetModuleInfoW64(h, addr, ctypes.byref(mi)) and mi.ModuleName:
                                lines.append(f"    {str(mi.ModuleName)}+0x{addr - mi.BaseOfImage:x}")
                            else:
                                lines.append(f"    0x{addr:x}")
                text = "\n".join(lines)
                if any(k in text for k in ("NtWait", "WaitFor", "Sleep", "ReadFile", "NtRead", "NtDeviceIo",
                                           "NtFsControl", "select")):
                    text = "  [wait]\n" + text
                out.append(text)
        finally:
            for _tid, th in handles:
                k32.ResumeThread(th)
                k32.CloseHandle(th)
        dbg.SymCleanup(h)
    finally:
        k32.CloseHandle(h)
    return out


def sample(pid: int) -> str:
    """Cheap: no suspend, no dump - each thread's state and wait reason, straight from the OS."""
    ps = ("$p = Get-Process -Id %d -ErrorAction SilentlyContinue; "
          "if ($p) { $p.Threads | ForEach-Object { "
          "'{0} {1} {2} {3}' -f $_.Id, $_.ThreadState, $_.WaitReason, "
          "[math]::Round($_.TotalProcessorTime.TotalMilliseconds) } }" % pid)
    p = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    if not lines:
        return f"(no threads read: {p.stderr.strip()[:200]})"
    waiting = sum(1 for ln in lines if "Wait" in ln)
    return f"{len(lines)} threads, {waiting} waiting\n" + "\n".join(lines[:40])


# ---------------------------------------------------------------- selftest
DUMMY = r'''
import os, socket, sys, threading, time
def sleeper():
    while True:
        time.sleep(1.0)
def pipe_reader():
    a, b = os.pipe()          # b is never written and never closed: read blocks in NtReadFile
    os.read(a, 16)
def event_waiter():
    threading.Event().wait()
def on_socket():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)
    c = socket.socket(); c.connect(s.getsockname())   # a connected socket with nothing to say
    c.recv(16)
for f in (sleeper, pipe_reader, event_waiter, on_socket):
    threading.Thread(target=f, daemon=True).start()
print("ready", flush=True)
time.sleep(120)
'''


def selftest() -> int:
    tmp = Path(os.environ.get("TEMP", ".")) / "pc_dump_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    script = tmp / "dummy.py"
    script.write_text(DUMMY, encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-u", str(script)], stdout=subprocess.PIPE, text=True)
    print("dummy:", p.pid, p.stdout.readline().strip())
    time.sleep(2.0)
    ok = True
    print("\n--- live stacks (StackWalk64 + PDB) ---")
    stacks = live_stacks(p.pid, exe_dir=str(Path(sys.executable).parent))
    text = "\n".join(stacks)
    waits = [ln for ln in text.splitlines() if any(k in ln for k in
             ("NtWait", "WaitFor", "Sleep", "ReadFile", "NtRead", "recv", "select", "poll"))]
    print(text[:4000])
    print(f"\nthreads with a wait frame: {len(waits)}")
    ok &= len(stacks) >= 5 and len(waits) >= 2
    print("\n--- cheap sample (no suspend) ---")
    print(sample(p.pid))
    dmp = tmp / "dummy.dmp"
    good, msg = write_dump(p.pid, dmp)
    print(f"\n--- dump: {msg} ---")
    ok &= good
    if good:
        print_dump(dmp)
        ok &= dmp.stat().st_size > 20_000
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
    try:
        p.stdout.close()
    except OSError:
        pass
    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    sys.stdout.flush()
    # dbghelp's callbacks outlive the interpreter's teardown on this path and fault after the last print (0xC0000005),
    # so the self-test leaves by the front door instead of letting the process unwind.
    os._exit(0 if ok else 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["dump", "stacks", "read", "sample", "selftest"])
    ap.add_argument("target", nargs="?", help="a pid (dump/stacks/sample) or a dump path (read)")
    ap.add_argument("--pid", type=int)
    ap.add_argument("--out", default="hung.dmp")
    ap.add_argument("--exe-dir", default="")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "read":
        print_dump(Path(a.target or a.out))
        return 0
    pid = a.pid or (int(a.target) if a.target and a.target.isdigit() else None)
    if not pid:
        ap.error("need --pid or a pid")
    if a.cmd == "dump":
        good, msg = write_dump(pid, Path(a.out))
        print(("wrote " if good else "failed: ") + msg)
        return 0 if good else 1
    if a.cmd == "stacks":
        print("\n".join(live_stacks(pid, exe_dir=a.exe_dir or str(Path(sys.executable).parent))))
        return 0
    print(sample(pid))
    return 0


if __name__ == "__main__":
    sys.exit(main())
