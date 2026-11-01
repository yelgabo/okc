"""Generate measurement-only matched prologue exits; external source stays outside git.

Usage: python3 tools/attn_make_prologue_probe.py DONOR.py OUT.py
The normal judged path remains our unmodified submission. Never score the output.
"""
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DONOR_SHA = "e6d929d40e6cff8aa82270a423fca24dfccad64b6a82428db9c3377b661bf124"


def main():
    donor, out = map(Path, sys.argv[1:])
    raw = donor.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DONOR_SHA
    src = raw.decode()

    def replace(old, new):
        nonlocal src
        assert src.count(old) == 1, (old, src.count(old))
        src = src.replace(old, new)

    replace("  unsigned ns_magic;", "  int phase_limit;\n  unsigned ns_magic;")
    replace("  p.coop = (int)((tma >> 1) & 1);",
            "  p.phase_limit = (int)(tma >> 20);\n  p.coop = (int)((tma >> 1) & 1);")
    replace("    if flat or cs <= 1:\n", "    tma |= (_DIAG_PHASE << 20)\n    if flat or cs <= 1:\n")

    def exit_before(phase, marker):
        replace(marker, f"  if constexpr (!FLAT) {{ if (p.phase_limit == {phase}) {{ cp_wait<0>(); return true; }} }}\n" + marker)

    replace("  cp_commit();\n  if (!WS && (MODE & 8)) {",
            "  cp_commit();\n  if constexpr (!FLAT) { if (p.phase_limit == 1) { cp_wait<0>(); return true; } }\n  if (!WS && (MODE & 8)) {")
    exit_before(2, "  // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----")
    replace("  csync<WS, NT>();\n  if constexpr (!FLAT && (MODE & 8) != 0) {",
            "  csync<WS, NT>();\n  if constexpr (!FLAT) { if (p.phase_limit == 3) { cp_wait<0>(); return true; } }\n  if constexpr (!FLAT && (MODE & 8) != 0) {")
    exit_before(5, "  // ---- 3. while they load: RoPE table, queries, the block's new keys / values ----")
    replace("  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's first barrier\n",
            "  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's first barrier\n  if constexpr (!FLAT) { if (p.phase_limit == 7) { cp_wait<0>(); return true; } }\n")
    exit_before(9, "  // ---- consumer state ----")
    # Runtime phase travels in unused launch-bitfield bits: all arms share one CUDA build.
    src += "\n_DIAG_PHASE = 0\n"
    patched = out.with_suffix(".donor_probe.py")
    patched.write_text(src)
    args = [sys.executable, str(ROOT / "tools/attn_make_bench_ab.py"), str(out),
            "--shapes", "b1", "--inputs", "3", "--rounds", "100", "--seed", "311101",
            "--null", "--pub", f"pubdiag={patched}"]
    for p in (1, 2, 3, 5, 7, 9):
        args += [f"p{p}_probe@own=-DONE_DIAG=1:phase={p}",
                 "--pubsub", f"pubp{p}_probe=pubdiag:_DIAG_PHASE = 0=>_DIAG_PHASE = {p}"]
    subprocess.run(args, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()

