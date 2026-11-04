import subprocess
import sys
import time

sys.path.insert(0, "tools")
import bx_box_run as bx  # noqa: E402

path = "/tmp/bx_clock_test.lock"
open(path, "a").close()
bx.BudgetClock.install()
lock = bx.GpuLock(path)
lock.acquire("start")
deadline = time.monotonic() + 15
wall0, file0 = bx.REAL_MONO(), time.monotonic()
lock.release("build stand-in")
holder = subprocess.Popen(["flock", path, "sleep", "20"])
time.sleep(1)
time.sleep(2)  # the stand-in build: counts against the budget, as on the judge
lock.acquire("build stand-in done")  # blocks about 17 s behind the other holder
holder.wait()
wall, seen = bx.REAL_MONO() - wall0, time.monotonic() - file0
print(f"real elapsed {wall:.1f} s, file clock elapsed {seen:.1f} s, offset {bx.BudgetClock.offset:.1f} s,"
      f" deadline (15 s budget) {'exceeded' if time.monotonic() >= deadline else 'not exceeded'}")
assert wall > 15 and seen < 15
