"""tools/minimax/test_pc_launch.py - the shared-PC launcher's lock and RAM rules, with no engine and no GPU.

Every case runs the launcher as its own process against a temp lock file (`--lock`), so what is tested is the real
file protocol the Maya task and this one share: take/release, a live owner, a stale PID, a bare-name lock, the RAM
floor, the watchdog, stdin passthrough and the "no BOM, plain ASCII" rule.

    python -m pytest tools/minimax/test_pc_launch.py -q
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LAUNCH = HERE / "pc_launch.py"
ROOT = HERE.parents[1]


# the lock cases must not trip the RAM gate: the box may be busy with the other task while they run
LOCK_ONLY = ["--est-ram-gb", "0.1", "--min-free-gb", "0", "--modified-source", "seq:0"]


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def run(args: list[str], stdin: str | None = None, timeout: float = 60,
        env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(LAUNCH), *args], input=stdin, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)


def read(lock: Path) -> str:
    return lock.read_text(encoding="ascii", errors="replace").strip()


def test_takes_and_releases(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    p = run(["--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--", *py("print('child ran')")])
    assert p.returncode == 0, p.stderr
    assert "child ran" in p.stdout
    assert read(lock) == "free"
    # plain ASCII, no BOM, and the release is exactly the four bytes the protocol names
    assert lock.read_bytes() == b"free"
    assert "lock is free, taking it for 'dspark'" in p.stderr


def test_lock_line_carries_owner_pid_and_time(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    # the child reads the lock while it is held
    code = ("import pathlib,sys;"
            f"print(pathlib.Path(r'{lock}').read_text('ascii'))")
    p = run(["--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--", *py(code)])
    assert p.returncode == 0, p.stderr
    lines = p.stdout.strip().splitlines()
    parts = lines[0].split()
    assert len(parts) == 3, lines[0]
    assert parts[0] == "dspark"
    assert parts[1].isdigit()
    assert "T" in parts[2]          # an ISO timestamp
    assert lines[1].startswith("tree ") and lines[1].split()[1].isdigit()
    assert b"\xef\xbb\xbf" not in lock.read_bytes()


def test_refuses_a_live_owner(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    # our own PID is alive by definition
    lock.write_text(f"maya {os.getpid()} 2026-10-25T15:00:00+03:00", encoding="ascii")
    p = run(["--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--", *py("print('must not run')")])
    assert p.returncode == 2, p.stderr
    assert "REFUSING" in p.stderr and "maya" in p.stderr
    assert "must not run" not in p.stdout
    assert read(lock).startswith("maya ")


def test_takes_over_a_stale_pid(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_text("maya 999999 2026-10-24T09:00:00+03:00", encoding="ascii")
    p = run(["--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--", *py("print('ran')")])
    assert p.returncode == 0, p.stderr
    assert "stale PID" in p.stderr
    assert "ran" in p.stdout
    assert read(lock) == "free"


def test_bare_name_counts_as_held(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"maya")   # exactly what the file looks like today
    p = run(["--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--", *py("print('must not run')")])
    assert p.returncode == 2, p.stderr
    assert "no PID on the line" in p.stderr
    assert "must not run" not in p.stdout
    assert read(lock) == "maya"


def test_refuses_when_ram_will_not_fit(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    marker = tmp_path / "ran.txt"
    p = run(["--owner", "dspark", "--lock", str(lock), "--est-ram-gb", "100000",
             "--modified-source", "seq:0", "--",
             *py(f"open(r'{marker}','w').write('x')")])
    assert p.returncode == 3, p.stderr
    assert "REFUSING" in p.stderr and "floor" in p.stderr
    assert not marker.exists()
    assert read(lock) == "free"


def test_estimate_reads_the_engine_flags():
    sys.path.insert(0, str(HERE))
    import pc_launch  # noqa: E402

    est, parts = pc_launch.estimate_ram_gb(
        ["build/strata-minimax.exe", "--model", "m.gguf", "--memory-limit-mib", "86000", "--max-context", "4096",
         "--minimax-kv", "ram", "--minimax-prefill-batch", "4096"], 128.0)
    assert abs(est - (86000 / 1024 + 2.0 + 4096 * pc_launch.KV_GB_PER_TOKEN + 0.7 + 7.0)) < 1e-6, parts
    assert any("pinned expert tier 84.0 GB" in p for p in parts)
    assert any("over-tier resident 7.0 GB" in p for p in parts)
    est2, _ = pc_launch.estimate_ram_gb(["x", "--minimax-ram-experts-per-layer", "40"], 128.0)
    assert est2 > 40 * 57 * pc_launch.EXPERT_GB
    est3, parts3 = pc_launch.estimate_ram_gb(
        ["x", "--memory-limit-mib", "86000", "--minimax-dense-layers-gpu", "0"], 128.0, over_tier_gb=10.0)
    assert abs(est3 - (86000 / 1024 + 12.0 + 0.1 + 0.1 + 10.0)) < 1e-6, parts3


def test_watchdog_stops_the_run(tmp_path: Path):
    """A scripted RAM sequence: 20 GB at the gate, then 3 GB - under the 10 GB floor - so the first later poll fires."""
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    p = run(["--owner", "dspark", "--lock", str(lock), "--ram-source", "seq:20,20,3,3", "--est-ram-gb", "1",
             "--modified-source", "seq:0", "--min-free-gb", "10", "--poll", "0.2", "--", *py("import time; time.sleep(30)")], timeout=90)
    assert p.returncode == 4, p.stderr
    assert "WATCHDOG" in p.stderr and "3.0 GB" in p.stderr
    assert read(lock) == "free"


def test_stdin_reaches_the_child(tmp_path: Path):
    """The engine's serve protocol reads GEN lines from stdin: the launcher must not swallow it."""
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    code = "import sys; line = sys.stdin.readline().strip(); print('got:' + line)"
    p = run(["--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--", *py(code)], stdin="GEN 3 temperature=0 1 2 3\n")
    assert p.returncode == 0, p.stderr
    assert "got:GEN 3 temperature=0 1 2 3" in p.stdout


def test_child_exit_code_passes_through(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    p = run(["--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--", *py("raise SystemExit(7)")])
    assert p.returncode == 7, p.stderr
    assert read(lock) == "free"


def test_two_launchers_one_wins(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    first = subprocess.Popen([sys.executable, str(LAUNCH), "--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--",
                              *py("import time; time.sleep(4)")],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # wait until the first one holds it
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if read(lock).startswith("dspark "):
                break
            time.sleep(0.05)
        assert read(lock).startswith("dspark ")
        second = run(["--owner", "maya", "--lock", str(lock), *LOCK_ONLY, "--", *py("print('must not run')")])
        assert second.returncode == 2, second.stderr
        assert "must not run" not in second.stdout
        assert read(lock).startswith("dspark ")   # the second launcher left the lock alone
    finally:
        first.communicate(timeout=60)
    assert first.returncode == 0
    assert read(lock) == "free"


def test_interrupt_releases(tmp_path: Path):
    """An interrupt on the launcher must stop the child and put the lock back to free.

    Windows has no SIGINT for another process: the launcher handles SIGBREAK and the test sends CTRL_BREAK_EVENT to
    its process group, which is what Ctrl-Break does.  Elsewhere it is a plain SIGINT.

    The child sleeps 30s and inherits this process's stdout/stderr pipes, so an orphaned child keeps `communicate`
    blocked for the rest of that sleep.  The elapsed-time check is what catches that: the run must return promptly,
    not merely end after the child would have exited by itself.
    """
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen([sys.executable, str(LAUNCH), "--owner", "dspark", "--lock", str(lock), "--poll", "0.2",
                             *LOCK_ONLY, "--", *py("import time; time.sleep(30)")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=flags)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not read(lock).startswith("dspark "):
            time.sleep(0.05)
        assert read(lock).startswith("dspark "), proc.stderr.read() if proc.poll() is not None else "no lock"
        if sys.platform == "win32":
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        t0 = time.monotonic()
        out, err = proc.communicate(timeout=60)
        elapsed = time.monotonic() - t0
        assert proc.returncode == 130, err
        # "before the child started" is legitimate: the interrupt may beat the spawn, which is exactly the race.
        assert ("stopping the child tree" in err or "child exited" in err
                or "stopping before the child started" in err), err
        assert elapsed < 15, f"the launcher took {elapsed:.1f}s: the child was left running\n{err}"
    finally:
        if proc.poll() is None:
            proc.kill()
    assert read(lock) == "free"


def test_the_lock_records_the_child_tree_and_who_prints_it(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    proc = subprocess.Popen([sys.executable, str(LAUNCH), "--owner", "dspark", "--lock", str(lock), *LOCK_ONLY,
                             "--", *py("import time; time.sleep(8)")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        tree = []
        while time.monotonic() < deadline:
            lines = read(lock).splitlines()
            if len(lines) > 1 and lines[1].startswith("tree "):
                tree = [int(x) for x in lines[1].split()[1:]]
                break
            time.sleep(0.05)
        assert tree, read(lock)
        assert all(pid > 0 for pid in tree)
        who = run(["--who", "--lock", str(lock)], timeout=30)
        assert who.returncode == 0, who.stderr
        assert "owner=dspark" in who.stdout
        assert "tree=" + " ".join(str(x) for x in tree) in who.stdout
    finally:
        proc.communicate(timeout=60)
    assert read(lock) == "free"


def test_dirty_pages_count_against_the_floor(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    marker = tmp_path / "ran.txt"
    p = run(["--owner", "dspark", "--lock", str(lock), "--ram-source", "seq:20",
             "--modified-source", "seq:15", "--est-ram-gb", "1", "--min-free-gb", "10", "--",
             *py(f"open(r'{marker}','w').write('x')")])
    assert p.returncode == 3, p.stderr
    assert "modified (dirty) pages" in p.stderr
    assert "15.0" in p.stderr and "effective" in p.stderr
    assert not marker.exists()
    assert read(lock) == "free"


def test_dirty_pages_are_logged_and_leave_enough_when_small(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    p = run(["--owner", "dspark", "--lock", str(lock), "--ram-source", "seq:20",
             "--modified-source", "seq:1", "--est-ram-gb", "1", "--min-free-gb", "10", "--",
             *py("print('ran')")])
    assert p.returncode == 0, p.stderr
    assert "ran" in p.stdout
    assert "20.0 GB available" in p.stderr and "1.0 GB in modified" in p.stderr


def test_watchdog_uses_effective_ram_after_dirty_pages(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    p = run(["--owner", "dspark", "--lock", str(lock), "--ram-source", "seq:20,20,20",
             "--modified-source", "seq:0,0,15", "--est-ram-gb", "1", "--min-free-gb", "10",
             "--poll", "0.2", "--", *py("import time; time.sleep(30)")], timeout=90)
    assert p.returncode == 4, p.stderr
    assert "WATCHDOG" in p.stderr and "modified 15.0" in p.stderr
    assert read(lock) == "free"


def test_turn_limit_stops_the_child_and_releases(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    p = run(["--owner", "dspark", "--lock", str(lock), *LOCK_ONLY, "--max-minutes", "0.03", "--poll", "0.05",
             "--", *py("import time; time.sleep(30)")], timeout=90)
    assert p.returncode == 6, p.stderr
    assert "TURN LIMIT" in p.stderr
    assert read(lock) == "free"


def test_calibration_records_peak_and_later_estimates_use_it(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    cal = tmp_path / "calibration.json"
    child = py("import time; time.sleep(0.5)")
    env = dict(os.environ, PC_LAUNCH_FAKE_PEAK_GB="20")
    base = ["--owner", "dspark", "--lock", str(lock), "--calibration", str(cal),
            "--ram-source", "seq:100", "--modified-source", "seq:0", "--est-ram-gb", "1"]
    p = run([*base, "--record-peak", "--", *child], env=env)
    assert p.returncode == 0, p.stderr
    assert "peak working set 20.0 GB" in p.stderr
    data = json.loads(cal.read_text(encoding="utf-8"))
    assert len(data) == 1 and next(iter(data.values()))["peak_working_set_gb"] == 20.0

    # Same child model+flags, a dry run with 25 GB effective: formula 1 GB, calibrated 20 GB -> 5 GB left < 10.
    p2 = run(["--owner", "dspark", "--lock", str(lock), "--calibration", str(cal),
              "--ram-source", "seq:25", "--modified-source", "seq:0", "--est-ram-gb", "1",
              "--min-free-gb", "10", "--dry-run", "--", *child])
    assert p2.returncode == 3, p2.stderr
    assert "calibrated peak 20.0 GB" in p2.stderr
    assert "formula 1.0 GB, using 20.0 GB" in p2.stderr

    # 30 GB effective leaves exactly the 10 GB floor: the calibrated estimate now lets it through.
    p3 = run(["--owner", "dspark", "--lock", str(lock), "--calibration", str(cal),
              "--ram-source", "seq:30", "--modified-source", "seq:0", "--est-ram-gb", "1",
              "--min-free-gb", "10", "--dry-run", "--", *child])
    assert p3.returncode == 0, p3.stderr
    assert "formula 1.0 GB, using 20.0 GB" in p3.stderr


def test_over_tier_flag_reaches_the_estimate(tmp_path: Path):
    lock = tmp_path / "pc-engine.lock"
    lock.write_bytes(b"free")
    p = run(["--owner", "dspark", "--lock", str(lock), "--over-tier-gb", "11",
             "--ram-source", "seq:100", "--modified-source", "seq:0", "--dry-run", "--",
             *py("print('no')"), "--model", "m.gguf", "--memory-limit-mib", "40960"])
    assert p.returncode == 0, p.stderr
    assert "over-tier resident 11.0 GB" in p.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
