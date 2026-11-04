#!/usr/bin/env python3
"""ly_v1fs: ly_v1un with a dynamic shared-memory reservation on the T = 8 finish launch (verify 1 finish spread).

Base: muse_decode_layer/cand/ly_v1un.py (SHA256 88fbae86..., checked). At T = 8 the finish
(_sl_cluster_finish_cg, Triton, 16 warps, 27 registers, 4,096 B of shared memory, 64 programs) is a PDL secondary
of the 128-CTA unpooler slice sk2 (118,240 B + 8 x pmax of shared memory per CTA). The v1st stamps (run log
"Verify 1 slice and finish (box)") put 14 of its 64 programs 3 or 4 per SM on the 4 SMs without a slice CTA; those
end 0.7 us after the programs that sit alone beside a slice CTA and set the finish's end.

The change: the compiled T = 8 finish kernels (num_warps 16, used only at T = 8; T = 1 runs 8 warps) launch with
SMEM bytes of dynamic shared memory instead of Triton's 4,096 (packed_metadata[2], plus the function's
MAX_DYNAMIC_SHARED_SIZE_BYTES attribute). The kernel's code, warps, grid and reduction order are unchanged, so the
SASS and the output are the base's. An SM has 233,472 B and every CTA reserves 1,024 B more:

- SMEM 118,784 (default, "one per SM"): 2 x 119,808 > 233,472, so one finish program per SM, and 119,808 + 119,264
  > 233,472, so none fits beside a slice CTA: 4 start on the free SMs, the other 60 as slice CTAs exit.
- SMEM 98,304 ("two per free SM"): 3 x 99,328 > 233,472 >= 2 x 99,328, and 99,328 + 119,264 + 8 x pmax fits for
  pmax <= 1,860: at most 2 per free SM, one beside a slice CTA.

`--env` (instrument builds only) reads the reservation from the V1FS_SMEM environment variable at import.

    python3 tools/dl_ly_v1fs_patch.py [--smem 118784] [--env] [--out muse_decode_layer/cand/ly_v1fs.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_v1fs.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


SMEM = int(arg("--smem", "118784"))
if "--out" in sys.argv:
    DST = ROOT / arg("--out", "")
ENV = "--env" in sys.argv
SMEM_EXPR = f"int(os.environ.get('V1FS_SMEM', '{SMEM}'))" if ENV else str(SMEM)

HELPER = f'''_V1FS_SMEM = {SMEM_EXPR}   # v1fs: dynamic shared memory of the T = 8 finish launch (tools/dl_ly_v1fs_patch.py)
_V1FS_CU = None


def _v1fs_reserve():
    """Every compiled T = 8 finish kernel (16 warps; T = 1 runs 8) launches with _V1FS_SMEM bytes of dynamic shared
    memory: the reservation decides how many finish programs an SM takes, alone or beside a slice sk2 CTA. Code,
    grid and reduction order stay the base's."""
    global _V1FS_CU
    cache = _sl_cluster_finish_cg.device_caches[triton.runtime.driver.active.get_current_device()][0]
    for k in cache.values():
        if k.metadata.num_warps != SL_CLUSTER_CONFIG[8][0] or k.packed_metadata[2] >= _V1FS_SMEM:
            continue
        k._init_handles()
        if _V1FS_CU is None:
            _V1FS_CU = _C.CDLL("libcuda.so.1")
        rc = _V1FS_CU.cuFuncSetAttribute(_C.c_void_p(k.function), _C.c_int(8), _C.c_int(_V1FS_SMEM))  # MAX_DYNAMIC_SHARED_SIZE_BYTES
        if rc != 0:
            raise RuntimeError(f"v1fs: cuFuncSetAttribute returned {{rc}}")
        k.packed_metadata = (k.packed_metadata[0], k.packed_metadata[1], _V1FS_SMEM)


def _pdl8_sl_tile_after_attention(data, state, o_in, gate, T):
'''
LAUNCH = ("    _sl_cluster_finish_cg[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,"
          "launch_pdl=_pdl8_edge(\"fin\"))\n")
EDITS = [
    ("def _pdl8_sl_tile_after_attention(data, state, o_in, gate, T):\n", HELPER),
    # before: kernels compiled by earlier calls; after: the one this call compiled, for the next call
    ("    nw,bc=SL_CLUSTER_CONFIG[T]\n" + LAUNCH,
     "    nw,bc=SL_CLUSTER_CONFIG[T]\n    if T == 8:\n        _v1fs_reserve()\n" + LAUNCH
     + "    if T == 8:\n        _v1fs_reserve()\n"),
]
HEADER = (f"# ly_v1fs: ly_v1un with {SMEM} B of dynamic shared memory on the T = 8 finish launch, so its programs land "
          "one per SM (tools/dl_ly_v1fs_patch.py)\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"anchor found {text.count(old)} times: {old[:80]!r}")
        text = text.replace(old, new)
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
