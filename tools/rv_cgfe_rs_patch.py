#!/usr/bin/env python3
"""cgfe on the public layer file: writes muse_decode_layer/cand/rv_cgfe_rs.py.

g1b4_cgfe.py (3785f13a) is the release with the m = 1 cgemv block (tools/dl_cg_patch.py, bytes 440397ab), then the
pdl8b block, then the block that makes the three cgemv kernels PDL secondaries (wait before x, x through __ldcg,
trigger at the CTA's end). g1b4_rs.py (01426b89) is the same release prefix, byte for byte, then the pdl8d block with
the routing-slack edits. The port keeps cgfe's prefix (release + cgemv), puts g1b4_rs's PDL tail in place of pdl8b's,
and appends cgfe's cgemv PDL block unchanged. Every split point is checked.

    python3 tools/rv_cgfe_rs_patch.py                      # onto g1b4_rs, writes rv_cgfe_rs.py
    python3 tools/rv_cgfe_rs_patch.py --base pdl8h         # onto g1b4_pdl8h (33c2353b), writes rv_cgfe_h.py
    python3 tools/rv_cgfe_rs_patch.py --base pdl8hn        # onto g1b4_pdl8hn (df5cded9, no routing slack), writes rv_cgfe_hn.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
C = ROOT / "muse_decode_layer/cand"
BASES = {"rs": ("g1b4_rs.py", "01426b891ea12af13e8f713c9ca3a9f47900a07798212e53f583ed0005ff7f07", "rv_cgfe_rs.py"),
         "pdl8h": ("g1b4_pdl8h.py", "33c2353b52de29c5", "rv_cgfe_h.py"),
         "pdl8hn": ("g1b4_pdl8hn.py", "df5cded96b8dc18a", "rv_cgfe_hn.py")}
base = BASES[sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "rs"]
PINS = {base[0]: base[1],
        "g1b4_cgfe.py": "3785f13a5ee3",
        "g1b4_pdl8b.py": ""}
MARK = "# Added by yelgabo, November 3, 2026 (tools/dl_pdl8_patch.py)"
CG_PDL = "# pdl8e: the three m = 1 cgemv kernels"

src = {}
for name, pin in PINS.items():
    b = (C / name).read_bytes()
    if not hashlib.sha256(b).hexdigest().startswith(pin):
        raise SystemExit(f"{name} is not {pin}")
    src[name] = b.decode()
rs, cf, pb = src[base[0]], src["g1b4_cgfe.py"], src["g1b4_pdl8b.py"]
for name, s in src.items():
    if s.count(MARK) != 1:
        raise SystemExit(f"{name}: PDL block marker found {s.count(MARK)} times")
if rs[:rs.index(MARK)] != pb[:pb.index(MARK)]:
    raise SystemExit(f"{base[0]}'s prefix is not the release prefix")
cf_tail = cf[cf.index(MARK):]
if cf_tail.count(CG_PDL) != 1 or cf_tail[:cf_tail.index(CG_PDL)].rstrip("\n") != pb[pb.index(MARK):].rstrip("\n"):
    raise SystemExit("g1b4_cgfe's tail is not pdl8b's block plus the cgemv PDL block")
head = "# okc: problem=muse_decode_layer\n"
out = (cf[:cf.index(MARK)].replace(head, head + f"# {base[2][:-3]}: g1b4_cgfe's release + cgemv prefix, {base[0][:-3]}'s PDL tail, "
                                   "cgfe's cgemv PDL block (tools/rv_cgfe_rs_patch.py).\n", 1)
       + rs[rs.index(MARK):].rstrip("\n") + "\n\n\n" + cf_tail[cf_tail.index(CG_PDL):])
dst = C / base[2]
dst.write_text(out)
py_compile.compile(str(dst), doraise=True)
print(dst.relative_to(ROOT), hashlib.sha256(out.encode()).hexdigest(), len(out))
