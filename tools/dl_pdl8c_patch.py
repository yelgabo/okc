#!/usr/bin/env python3
"""Composed candidate: the public layer release (ba9e327c) + the parity lane's m = 1 cgemv block
(tools/dl_cg_patch.py, its check-2 bytes 440397ab) [+ its T = 1 split-front block with --front] + the pdl8b block
(tools/dl_pdl8b_patch.py, public fefd7f7b8e834bd7). Writes muse_decode_layer/cand/g1b4_pdl8c.py (g1b4_pdl8cf.py with
--front).

    python3 tools/dl_pdl8c_patch.py CG_TOOL C2 [--front]
      CG_TOOL  tools/dl_cg_patch.py from branch worktree-agent-a51254a802fef2de0
      C2       git show 3044ee7:muse_bf16_proj/submission.py (b2c4c096)

Order: the cgemv block comes first, so its prepare (tuner picks re-applied after the pinned-pick prepare) runs
inside the pdl8b prepare, and the PDL A/B and stress time the composed T = 1 chain. The cgemv kernels are plain
launches with no griddepcontrol, so the edges into them stay serialized and the edges out of them start at their
exit; the PDL enable map is set by the pdl8b prepare after every cgemv pick is final.
"""
import hashlib
import importlib.util
import pathlib
import py_compile
import sys

HERE = pathlib.Path(__file__).resolve().parent
CG_BYTES_SHA = "440397ab59440dd053c93d152f2110f3548ad97694615783a582fd6ced042e5a"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    front = "--front" in sys.argv[1:]
    cg_tool, c2 = args
    CG = load("dl_cg_patch", cg_tool)
    B = load("dl_pdl8b_patch", HERE / "dl_pdl8b_patch.py")
    release, block = B.make_block()
    rel_bytes = release.encode()
    c2_bytes = pathlib.Path(c2).read_bytes()
    plain = CG.build(rel_bytes, c2_bytes, front=False)
    if hashlib.sha256(plain).hexdigest() != CG_BYTES_SHA:
        sys.exit("the cgemv tool no longer reproduces the parity lane's check-2 bytes 440397ab")
    composed = CG.build(rel_bytes, c2_bytes, front=front).decode()
    for name in ("_pdl8", "PDL8_", "PDL_CHAIN"):
        if name in composed:
            sys.exit(f"{name} already present before the pdl8b block")
    head = "# okc: problem=muse_decode_layer\n"
    composed = composed.replace(head, head + "# pdl8c: release + cgemv m = 1 port" + (" + split T = 1 front" if front else "")
                                + " (tools/dl_cg_patch.py) + pdl8b (tools/dl_pdl8b_patch.py), composed by"
                                " tools/dl_pdl8c_patch.py.\n", 1)
    out = composed.rstrip("\n") + "\n" + block
    dst = B.A.ROOT / ("muse_decode_layer/cand/g1b4_pdl8cf.py" if front else "muse_decode_layer/cand/g1b4_pdl8c.py")
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst, hashlib.sha256(dst.read_bytes()).hexdigest(), len(dst.read_bytes()))


if __name__ == "__main__":
    main()
