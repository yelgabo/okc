"""Port projection candidate C1 (muse_bf16_proj 05286d9, standalone board entry
e1d3d341549c484d) into an M1 layer file such as layer_m1rt51s:

1. tools/dl_proj_inline.py with the standalone at the given path writes _SKP_SRC,
   _SKP_GEMV_SRC, _SKP_CPP_SRC, _SKP_WD_SRC, _SKP_CONFIGS and the @SKP_PY@ region. C1's sk
   changes arrive with it: NOVOTE (bit 1024 of NACC) on every compressed config and the FULL
   twins (name + "F", NACC bit 32) for M == TOKR on all of them but qkv32.
2. The inliner copies the standalone's sk source, which lacks the layer's M1 patches, so they
   are applied again from tools/dl_gate_patch.py unchanged: the reader-side
   fence.proxy.async.global after griddepcontrol.wait in thread 0 and in the producer lane
   (FENCE, PRODUCER) and the runtime PDL issue order in bit 30 of M (RUNTIME_REPS).
3. The layer's sk runner launches the FULL twin when the call has exactly TOKR tokens, as the
   standalone's _sk_runner does; the PDL bit and the census count are unchanged.

C1's third piece, the pooler m = 128 cuBLASLt pin, is not ported: the layer's 128-token front
is its own concat GEMM, not the standalone's Lt plan (docs/proj-port-guide.md).

    python3 tools/dl_proj_c1_port.py PROJ_SUBMISSION.py IN_LAYER.py OUT_LAYER.py "HEADER LINE 2"
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dl_gate_patch as gp  # noqa: E402

RUNNER_OLD = '''def _skp_runner(ext, name, stream, tab, pmax, base4, n, k, diag):
    cid = SK_INDEX[name]'''
RUNNER_NEW_HEAD = '''# Configs whose FULL twin has launched (C1); the SUMMARY lists them, so a self-test that never
# reached the twins shows.
_SKP_FULL_SEEN = set()


def _skp_runner(ext, name, stream, tab, pmax, base4, n, k, diag):
    cid = SK_INDEX[name]'''
RUNNER_OLD = RUNNER_OLD + '''

    def run(x, pdl=False):'''
RUNNER_NEW = RUNNER_NEW_HEAD + '''
    # Projection C1: a compressed config's FULL twin (compile-time full-tile x addressing)
    # serves the calls with exactly TOKR tokens, as the standalone's _sk_runner dispatches.
    full = SK_INDEX.get(name + "F")
    tokr = _SKP_CONFIGS[cid][2]

    def run(x, pdl=False):'''
LAUNCH_OLD = '''        ext.sk_fwd(cid | (SK_PDL_BIT if pdl else 0), stream, x, out, tab, pmax, base4, n, k, diag)'''
LAUNCH_NEW = '''        c = full if full is not None and x.shape[0] == tokr else cid
        if c != cid:
            _SKP_FULL_SEEN.add(name)
        ext.sk_fwd(c | (SK_PDL_BIT if pdl else 0), stream, x, out, tab, pmax, base4, n, k, diag)'''
SUMMARY_OLD = '''    for key in ("tune m=1", "tune m=8"):
        lines.append(f"skp {key}: " + "; ".join(_SKP_REPORT.get(key, [])))'''
SUMMARY_NEW = SUMMARY_OLD + '''
    lines.append("skp full twins launched: " + (",".join(sorted(_SKP_FULL_SEEN)) or "none"))'''


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    proj, src, out, header = sys.argv[1:]
    shutil.copyfile(src, out)
    subprocess.run([sys.executable, str(HERE / "dl_proj_inline.py"), proj, out], check=True)
    text = open(out).read()
    reps = [(gp.FENCE_OLD, gp.FENCE_NEW, 1), (gp.PRODUCER_OLD, gp.PRODUCER_NEW, 1)]
    # C1's FULL twins assign M = TOKR in the kernel, so the decoded M stays mutable; the rest
    # of the runtime-order patch applies as written.
    full_old = ("    using C = SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, FUSE>;\n"
                "    if constexpr (C::FULL) {\n        M = TOKR;\n    }\n")
    full_new = ("    using C = SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, FUSE>;\n"
                "    int M = Mf & 0x3fffffff;\n"
                "    if constexpr (C::FULL) {\n        M = TOKR;\n    }\n")
    runtime = list(gp.RUNTIME_REPS)
    if "const int M = Mf & 0x3fffffff;" not in runtime[1][1]:
        raise SystemExit("dl_gate_patch.RUNTIME_REPS changed; revisit the C1 port")
    runtime[1] = (full_old, full_new, 1)
    reps += runtime
    reps += [(RUNNER_OLD, RUNNER_NEW, 1), (LAUNCH_OLD, LAUNCH_NEW, 1),
             (SUMMARY_OLD, SUMMARY_NEW, 1)]
    for old, new, n in reps:
        if text.count(old) != n:
            raise SystemExit(f"expected {n} of {old[:70]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    if "SK_INDEX.get(name + \"F\")" not in text or "pdl_run" not in text:
        raise SystemExit("port incomplete")
    lines = text.split("\n")
    if lines[0] != "# okc: problem=muse_decode_layer":
        raise SystemExit("line 1 is not the okc header")
    lines[1] = header
    open(out, "w").write("\n".join(lines))
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
