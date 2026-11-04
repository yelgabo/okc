#!/usr/bin/env python3
"""Run-time claim of the last phase B units in the layer's TINY2 MoE build (verify 1, 8 tokens).

    python3 tools/dl_mo_claim_patch.py          # muse_decode_layer/cand/mo_claim.py from rv_k_combo2.py (7f0a3e36)
    python3 tools/dl_mo_claim_patch.py --tl     # muse_decode_layer/cand/mo_claim_tl.py: the same on the timeline
                                                # build mo_tl8.py (tools/dl_moe_tl.py), for the finish-spread stamps

The T = 8 timeline ("T = 8 MoE floor and gap", docs/runs-moe.md) puts 2.8 us of the kernel's tail in the spread
of the CTAs' phase B end times. With OKC_CLM=1 (TINY2 only) the WS path splits only the first nB - P phase B units
statically (the same muldiv split as before, over fewer units), and the last P = min(2G, nB - 4G) units (none
when nB < 6G) form a pool. When thread 0 refills a ring slot past the CTA's static sequence, it uses a ticket from
p.ctr[1] that it took (one atomicAdd) at the previous refill. A ticket below P names the pool unit nB - P + ticket, which goes into that slot
exactly as a static unit would. The first ticket at or past P, or the CTA's 32nd claim, ends the CTA's sequence,
and its phase B count is published in smem (S.wnB). Every refill attempts exactly one sequence index, OKC_NS
ahead of its use, so the compute warps (two CTA barriers later) and the helper warp (after its mbarrier wait on
an earlier unit) see a claim's result before they need it. p.ctr[1] is the existing per-launch pool counter: this
build never used it, and the kernel itself zeroes it after its grid barrier for the next launch, as it does for
the dynamic pools of the other builds. No host memset, and no new PDL edge. Units before the pool keep the static
assignment. The grid barrier, every fence, the doneA dependency and the final pass are unchanged. All edits are
under #if OKC_CLM, so TINY and MID compile to the same code as before.
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAND = ROOT / "muse_decode_layer/cand"
BASES = {False: ("rv_k_combo2.py", "7f0a3e360f085c4f21c1b760d10eaf251d3d379464a1f0556d52828cf92d5f37", "mo_claim.py"),
         True: ("mo_tl8.py", "7c878d4e46e9a90c765799af3c7d313890e9ecdada2cf00218bb1e2123487566", "mo_claim_tl.py")}
MARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
TINY2_ADD = 'MOE_TINY2_DEFINES += ("@moe_helper0",)\n'

EDITS = [
    # smem: claimed units, this CTA's phase B count once known, pool size
    ("  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges\n",
     "  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges\n"
     "#ifndef OKC_CLM\n#define OKC_CLM 0\n#endif\n"
     "#if OKC_CLM\n"
     "  int wcl[32];                         // CLM: phase B pool units this CTA claimed, in claim order\n"
     "  int wnB;                             // CLM: this CTA's phase B unit count once its claims ended (CLM_BIG before)\n"
     "  int wpool;                           // CLM: pool size P\n"
     "#endif\n"),
    # helpers, before the WS loop
    ("__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\n",
     "__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\n"
     "#if OKC_CLM\n"
     "static_assert(OKC_WS && OKC_WSB && OKC_WSBLK <= 1 && OKC_NS >= 2, \"CLM: WS build, one compute-warp epilogue, ring of 2+\");\n"
     "#define CLM_BIG (1 << 30)\n"
     "// pool size: none unless every CTA keeps at least 4 static phase B units (then no claim falls in the preissue)\n"
     "__device__ __forceinline__ int clm_pool(int nB, int G) { const int a = 2 * G, b = nB - 4 * G; return b > 0 ? (a < b ? a : b) : 0; }\n"
     "__device__ __forceinline__ int clm_nb(const Smem& S) { return *reinterpret_cast<volatile const int*>(&S.wnB); }\n"
     "// phase B index ib of this CTA exists: static, or a claim that succeeded (the attempt for ib is OKC_NS units old)\n"
     "__device__ __forceinline__ bool clm_has(const Smem& S, int nBs, int ib) { return ib < nBs || ib < clm_nb(S); }\n"
     "// thread 0, refilling the slot of sequence index N + k (k-th index past the static sequence) with the ticket it\n"
     "// took one refill earlier: the atomic's round trip overlaps a unit's compute instead of holding the CTA barrier.\n"
     "// Every ticket taken is used at the next refill (a CTA holding one always has a next refill), so no pool unit is\n"
     "// lost; the 32nd claim takes no further ticket, which ends the CTA's sequence.\n"
     "__device__ __forceinline__ void clm_claim(const Params& p, Smem& S, int k, int N, int nBs, unsigned char* ring, int& tk) {\n"
     "  if (clm_nb(S) != CLM_BIG) return;\n"
     "  if (tk < S.wpool) {\n"
     "    const int unit = S.njS * RB_B - S.wpool + tk, sl = (N + k) % OKC_NS;\n"
     "    S.wcl[k] = unit;\n"
     "    ring_issue<true>(p, S, unit, ring + sl * UNIT_BYTES, S.bars + sl);\n"
     "    tk = k + 1 < 32 ? atomicAdd(p.ctr + 1, 1) : CLM_BIG;\n"
     "  } else {\n"
     "    *reinterpret_cast<volatile int*>(&S.wnB) = nBs + k;\n"
     "  }\n"
     "}\n"
     "#endif\n"),
    # prologue: the static split covers nB - P units
    ("    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n",
     "#if OKC_CLM\n"
     "    const int P = clm_pool(nB, G), nBs = nB - P;\n"
     "    S.wsr[2] = muldiv_g(nBs, c, G, rG); S.wsr[3] = muldiv_g(nBs, c + 1, G, rG);\n"
     "    S.wpool = P;\n"
     "    S.wnB = P > 0 ? CLM_BIG : S.wsr[3] - S.wsr[2];\n"
     "#else\n"
     "    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n"
     "#endif\n"),
    # the compute warps' phase B range is the static one
    ("  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n",
     "#if OKC_CLM\n"
     "  skinny_ring_ws<PB>(p, S, PB ? S.wsr[2] : v0, PB ? S.wsr[3] : v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n"
     "#else\n"
     "  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n"
     "#endif\n"),
    # thread 0's ticket in hand (taken one refill before its use)
    ("  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)\n",
     "  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)\n"
     "#if OKC_CLM\n"
     "  int clm_tk = CLM_BIG;\n"
     "#endif\n"),
    # compute loop: static units, then claimed ones
    ("  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n",
     "#if OKC_CLM\n"
     "  for (int i = 0; PB ? clm_has(S, nU, i) : i < nU; ++i) {\n"
     "    const int v = (!PB || i < nU) ? v0 + i : S.wcl[i - nU];\n"
     "#else\n"
     "  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n"
     "#endif\n"),
    ("    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {\n",
     "#if OKC_CLM\n"
     "    if (PB && warp < OKC_WSBLK && !clm_has(S, nU, i + 1)) {\n"
     "#else\n"
     "    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {\n"
     "#endif\n"),
    ("      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n"
     "      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n",
     "#if OKC_CLM\n"
     "      if (q + OKC_NS < N) {\n"
     "        ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n"
     "        if (q + OKC_NS == N - 1 && S.wpool > 0) clm_tk = atomicAdd(p.ctr + 1, 1);  // the first ticket, one refill early\n"
     "      } else {\n"
     "        clm_claim(p, S, q + OKC_NS - N, N, N - nA, ring, clm_tk);\n"
     "      }\n"
     "      if (!(OKC_WSBL && PB && !clm_has(S, nU, i + 1))) mbar_arrive1(S.wsrf + bb);   // WSBL: the last unit stays with warp 0\n"
     "#else\n"
     "      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n"
     "      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n"
     "#endif\n"),
    # the CTA's last unit, run by warp 0 after the loop
    ("  if (PB && warp < OKC_WSBLK && warp < nU) {",
     "#if OKC_CLM\n"
     "  const int nT = PB ? clm_nb(S) : nU;  // known: the loop ended on a failed claim (or P = 0: the static count)\n"
     "#else\n"
     "  const int nT = nU;\n"
     "#endif\n"
     "  if (PB && warp < OKC_WSBLK && warp < nT) {"),
    ("    const int ul = nU - 1 - warp;\n", "    const int ul = nT - 1 - warp;\n"),
    ("    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n",
     "#if OKC_CLM\n"
     "    const int v = ul < nU ? v0 + ul : S.wcl[ul - nU], j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n"
     "#else\n"
     "    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n"
     "#endif\n"),
    ("  seq = seq0 + nU;\n", "  seq = seq0 + nT;\n"),
    # helper warp: epilogues of every unit but the CTA's last, claimed ones included
    ("  for (int q = 0; q < Nh; ++q) {\n",
     "#if OKC_CLM\n"
     "  const bool clm = S.wpool > 0;\n"
     "  for (int q = 0; clm ? (q + 1 < nA || clm_has(S, b1 - b0, q + 1 - nA)) : q < Nh; ++q) {\n"
     "#else\n"
     "  for (int q = 0; q < Nh; ++q) {\n"
     "#endif\n"),
    ("    const int v = PB ? b0 + (q - nA) : a0 + q;\n",
     "#if OKC_CLM\n"
     "    const int v = PB ? (q - nA < b1 - b0 ? b0 + (q - nA) : S.wcl[q - N]) : a0 + q;\n"
     "#else\n"
     "    const int v = PB ? b0 + (q - nA) : a0 + q;\n"
     "#endif\n"),
]

TL_EDITS = [
    ("      if (i == nU - 1) TLW(PB ? 6 : 3, t);\n",
     "#if OKC_CLM\n"
     "      if (PB ? !clm_has(S, nU, i + 1) : i == nU - 1) TLW(PB ? 6 : 3, t);\n"
     "#else\n"
     "      if (i == nU - 1) TLW(PB ? 6 : 3, t);\n"
     "#endif\n"),
    ("nB = (unsigned long long)(S.wsr[3] - S.wsr[2]);",
     "nB = (unsigned long long)(OKC_CLM ? clm_nb(S) : S.wsr[3] - S.wsr[2]);"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", action="store_true")
    a = ap.parse_args()
    src_name, pin, dst_name = BASES[a.tl]
    raw = (CAND / src_name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        sys.exit(f"{src_name} is not {pin[:8]}")
    whole = raw.decode()
    assert whole.count(MARK) == 1
    head, rest = whole.split(MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    # the loop edits belong to skinny_ring_ws only (skinny_ring, the non-WS loop, has the same lines)
    for old, new in EDITS + (TL_EDITS if a.tl else []):
        n = helper.count(old)
        if n == 1:
            helper = helper.replace(old, new)
            continue
        w1 = helper.index("__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {")
        w0 = helper.rindex("__device__ void skinny_ring_ws(", 0, w1)
        if helper[w0:w1].count(old) != 1:
            sys.exit(f"helper0 anchor found {n} times, {helper[w0:w1].count(old)} in skinny_ring_ws: {old[:90]!r}")
        helper = helper[:w0] + helper[w0:w1].replace(old, new) + helper[w1:]
    if tail.count(TINY2_ADD) != 1:
        sys.exit("TINY2 define anchor")
    tail = tail.replace(TINY2_ADD, TINY2_ADD + 'MOE_TINY2_DEFINES += ("-DOKC_CLM=1",)\n')
    out = head + MARK + helper + tail
    dst = CAND / dst_name
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
