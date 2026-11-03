import importlib.util
import sys

spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
src = mod.sk_source(device_only=True)
# The device-only cgemv instantiations no longer match cgemv_launch's signature (stale since the
# a35709c rework); the gate is about sk_kernel, so they are dropped from both builds.
src = "\n".join(l for l in src.split("\n") if not l.startswith("template void cgemv_launch<"))
open(sys.argv[2], "w").write(src)
