#!/usr/bin/env python3
"""ly_aqf: the q_len 8 one-tile attention (verify 1 sliding, attn_kernel<8, false, 15>) issues its tile's K/V copies
after the q pass instead of before it, on ly_stk1.

Round 2 stamps: thread 0's q pass (two iterations of register math on shared memory) reads 1.0 us after the wait,
while the same out-of-line code run before griddepcontrol.wait takes no visible time (ly_aqw), and neither the
fence after it (stamp before the fence: same time), the pooler bulk prefetch (ly_anpf) nor the instruction cache
(ly_aqw, ly_akw) explains it. Hypothesis: its shared-memory loads queue behind the 1,024 cp.async K/V copies issued
just before (4 per thread, in flight to DRAM). One change, behind constexpr QF = OKC_QFIRST8 && ONE && L == 8 && !S2:
the one-tile straight-line K/V issue (unchanged text, same commits, same group order: page indices, q, K/V) moves from
before the q group's wait to right after the q pass's fence.proxy.async; the q group's wait becomes cp.async.wait_group 0
(the K/V groups are not committed yet). The consumers' K/V wait, the new-row patch, every fence, the wait and the
trigger are unchanged.

    python3 tools/dl_ly_aqf_patch.py [--src muse_decode_layer/cand/ly_stk1.py] [--out muse_decode_layer/cand/ly_aqf.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / "muse_decode_layer/cand/ly_aqf.py"
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

ISSUE_START = "      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);\n"
ISSUE_END = "#pragma unroll\n      for (int k = 0; k < STAGES - 1; ++k) cp_commit();\n"
QFENCE = ("  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's "
          "first barrier\n  DBG(10);\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    blk = text[i:j]
    a = blk.index(ISSUE_START)
    b = blk.index(ISSUE_END, a) + len(ISSUE_END)
    issue = blk[a:b]
    late = ("  if constexpr (QF) {   // the tile's K/V copies, issued after the q pass (same text and commits as the ONE "
            "issue)\n    if (qf_one) {\n" + "".join("    " + ln.lstrip(" ") + "\n" if ln.lstrip().startswith("#") else ln + "\n"
                    for ln in issue[:-1].split("\n") if not ln.lstrip().startswith("//"))
            + "    }\n  }\n")
    edits = [
        ("#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n",
         "#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n"
         "#define OKC_QFIRST8 1   // q_len 8 one-tile builds: the tile's K/V copies issued after the q pass\n"),
        ("  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n",
         "  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n"
         "  constexpr bool QF = OKC_QFIRST8 && ONE && L == 8 && !Cfg<L, FLAT>::S2;\n"
         "  bool qf_one = false;\n"),
        ("      one = true;\n" + ISSUE_START,
         "      one = true;\n      if constexpr (QF) {\n        qf_one = true;\n      } else {\n" + ISSUE_START),
        (ISSUE_END + "    }\n    if (!one) {", ISSUE_END + "      }\n    }\n    if (!one) {"),
        ("    cp_wait<STAGES - 1>();\n    csync<WS, NT>();\n  }\n",
         "    if constexpr (QF) cp_wait<0>();   // (QF: no K/V group committed yet)\n    else cp_wait<STAGES - 1>();\n"
         "    csync<WS, NT>();\n  }\n"),
        (QFENCE, QFENCE + late),
    ]
    for old, new in edits:
        blk = sub(blk, old, new, 1, "att7")
    text = ("# ly_aqf: ly_stk1 + the q_len 8 one-tile attention issues its K/V copies after the q pass "
            "(tools/dl_ly_aqf_patch.py).\n") + text[:i] + blk + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
