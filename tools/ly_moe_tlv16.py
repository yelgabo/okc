#!/usr/bin/env python3
"""Verify 16 / 32 MAIN-build timeline: writes muse_decode_layer/cand/ly_fabq_tlv16.py from ly_fabq.py (3765470032).

tools/ly_moe_tlb4.py's level-1 stamps (per-CTA globaltimer at the kernel's phase points, the weights producer's issue
time of every stage) applied to the MAIN WGS build (M > 32: verify 16 and verify 32) instead of B4, with 336 slots per
CTA so that every stage of a verify 32 CTA is stamped (24 + n, n < 296), clock64 beside the globaltimer at routing
done / phase A end / phase B end / producers released (slots 320..323, the SM clock), and CTA 0's job table after the
stamps (p.dbg[136 * 336 + j] = jC | jE << 8 | jS << 16 | jrow << 32, [136 * 336 + 510] = njF, [511] = nj). The stamps
run only when a launch passes a fourth scratch tensor; the layer's own launches pass three. No fence, barrier,
mbarrier protocol, ring, flag or final-pass change; the consumers' stage loop is the control's text (level 1).

    python3 tools/ly_moe_tlv16.py [--base F --pin SHA] [--out muse_decode_layer/cand/ly_fabq_tlv16.py]
"""
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ly_moe_tlb4 as tb  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
DST = ROOT / "muse_decode_layer/cand/ly_fabq_tlv16.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
SLOTS = 336
MAIN = 'key, defines = "main", _main_defines(sfx)'
PUBMAIN = 'tuple(_main_defines(sfx)) + ("-DOKC_PUB=1",)'
EXTRAMAIN = 'tuple(_main_defines(sfx)) + MAIN_EXTRA_DEFINES'

EXTRA = [
    ("#else\n#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\n#endif\n",
     "#else\n#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\n"
     "#define TLC(k) do { } while (0)\n#endif\n"),
    ("#define DBG(k) do { if (TLB_ON && threadIdx.x == 0) p.dbg[blockIdx.x * 336 + (k)] = gtimer(); } while (0)\n",
     "#define DBG(k) do { if (TLB_ON && threadIdx.x == 0) p.dbg[blockIdx.x * 336 + (k)] = gtimer(); } while (0)\n"
     "#define TLC(k) do { if (TLB_ON && threadIdx.x == 0) p.dbg[blockIdx.x * 336 + (k)] = (unsigned long long)clock64(); } while (0)\n"),
    ("  DBG(1);\n#if OKC_MFIN && OKC_WGS && (OKC_MFQ & 8)\n", "  DBG(1);\n  TLC(320);\n#if OKC_MFIN && OKC_WGS && (OKC_MFQ & 8)\n"),
    ("    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n    DBG(2);\n",
     "    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n    DBG(2);\n    TLC(321);\n"),
    ("    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    DBG(3);\n",
     "    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    DBG(3);\n    TLC(322);\n"
     "#if OKC_TLB\n    if (TLB_ON && tid == 0 && blockIdx.x == 0) {\n"
     "      const int nj = S.njF + S.njS;\n"
     "      for (int j = 0; j < nj && j < JMAX; ++j)\n"
     "        p.dbg[136 * 336 + j] = (unsigned long long)S.jC[j] | ((unsigned long long)S.jE[j] << 8) |\n"
     "                               ((unsigned long long)S.jS[j] << 16) | ((unsigned long long)S.jrow[j] << 32);\n"
     "      p.dbg[136 * 336 + 510] = (unsigned long long)S.njF;\n"
     "      p.dbg[136 * 336 + 511] = (unsigned long long)nj;\n"
     "    }\n#endif\n"),
    ("  WS_JOIN();\n  DBG(6);\n", "  WS_JOIN();\n  DBG(6);\n  TLC(323);\n"),
]


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    whole = raw.decode()
    assert whole.count(tb.START) == 1 and whole.count(tb.END) == 1
    head, rest = whole.split(tb.START)
    cu, tail = rest.split(tb.END)
    edits = [(o, n.replace("blockIdx.x * 64", f"blockIdx.x * {SLOTS}").replace("n < 40", "n < 296")) for o, n in tb.EDITS]
    # ly_pub bases: the phase calls carry the publish count and the sentinel sits before DBG(3)
    PHA = "    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n"
    PHB = "    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    DBG(3);\n"
    PHA_PUB = "    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq, npub);\n"
    PHB_PUB = ("    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq, npub);\n#if OKC_PUB\n"
               "    wgs_publish(S, WARP_IDX(tid) >> 2, tid & 127, npub, nullptr, 0);  // the publisher may exit\n#endif\n    DBG(3);\n")
    if PHB_PUB in cu:
        edits = [(o.replace(PHA, PHA_PUB).replace(PHB, PHB_PUB), n.replace(PHA, PHA_PUB).replace(PHB, PHB_PUB)) for o, n in edits]
        extra = [(o.replace(PHA, PHA_PUB).replace(PHB, PHB_PUB), n.replace(PHA, PHA_PUB).replace(PHB, PHB_PUB)) for o, n in EXTRA]
    else:
        extra = EXTRA
    for old, new in edits + extra:
        n = cu.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, 1 expected: {old[:100]!r}")
        cu = cu.replace(old, new)
    if tail.count(MAIN) == 1:
        tail = tail.replace(MAIN, 'key, defines = "main", tuple(_main_defines(sfx)) + ("-DOKC_TLB=1",)')
    elif tail.count(PUBMAIN) == 1:  # ly_pub bases: the MAIN build already carries its own define
        tail = tail.replace(PUBMAIN, 'tuple(_main_defines(sfx)) + ("-DOKC_PUB=1", "-DOKC_TLB=1",)')
    elif tail.count(EXTRAMAIN) == 1:  # ly_rfin bases: the MAIN build's defines are a module tuple
        tail = tail.replace(EXTRAMAIN, EXTRAMAIN + ' + ("-DOKC_TLB=1",)')
    else:
        sys.exit("MAIN define anchor")
    out = head + tb.START + cu + tb.END + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
