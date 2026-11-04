#!/usr/bin/env python3
"""ly_akv: the q_len 1 one-tile attention build (ONE, page size 1) issues its cached K/V rows before griddepcontrol.wait.

Base ly_v1un (88fbae86). Only the per-request q_len 1 ONE builds change (attn_kernel<1, false, 5 / 7 / 13 / 15>;
decode 1 sliding runs 13): when the CTA takes the straight-line one-tile path (32-bit offsets, page size 1), it waits
for its page-index group, checks the speculated length, and issues the cached rows' K/V copies before
griddepcontrol.wait instead of after it. Every operand of those copies is a layer input (page_table, seq_lens, the
cached rows of k_cache / v_cache at positions below seq); the new token's row (position seq, slot cache_loc) is
never read from the cache (it is skipped, as today, and patched in from the q / k / v pass after the wait, which
reads qkv). q, the new k / v, the gate and every global write stay after the wait. After the wait the q group is the
newest cp.async group, so the q pass waits for all groups (cp.async.wait_group 0) instead of all but the K/V ones.
No fence, barrier, trigger or launch attribute moves; q_len 8 and the flat / coop builds compile as before
(OKC_PREKV gates on L == 1 and the ONE bit).

    python3 tools/dl_ly_akv_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--out muse_decode_layer/cand/ly_akv.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_akv.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

PRE = r"""    partL = tidL & 7;
  }
  bool pre = false;
  if constexpr (OKC_PREKV && L == 1 && ONE && !WS && !Cfg<L, FLAT>::S2 && OKC_ONEPRED) {
    // ly_akv: page_table, seq_lens and the cached K/V rows are layer inputs (no kernel of this step writes them; the
    // new token's row is skipped here and patched in after the wait from qkv)
    if (p.off32 && p.lp == 0) {
      cp_wait<0>();
      csync<WS, NT>();
      if constexpr (!FLAT && (MODE & 8) != 0) {
        if (seq_real_ != seq) {   // (speculated length missed: nothing issued yet, the caller re-runs)
          __syncthreads();
          return false;
        }
      }
      DBG(1);
      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
      const int cend = min(end, seq);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        const int pr = start + r;
        const bool cached = !OKC_ONENEW || pr < cend;
        const bool doit = cached || pr >= end;
        const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;
        cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);
        cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);
      }
#pragma unroll
      for (int k = 0; k < STAGES - 1; ++k) cp_commit();
      pre = true;
    }
  }

  pdl_wait();"""

EDITS = [
    ("#define OKC_PDLKV 0", "#define OKC_PREKV 1     // ly_akv: q_len 1 ONE builds issue the cached K/V rows before griddepcontrol.wait\n#define OKC_PDLKV 0", 1),
    ("    partL = tidL & 7;\n  }\n\n  pdl_wait();", PRE, 1),
    ("  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n", "  if (!pre) {\n  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n", 1),
    ("  DBG(8);\n  }\n", "  }   // !pre\n  DBG(8);\n  }\n", 1),
    ("  if constexpr (IDXF) {   // the q group",
     "  if (pre) cp_wait<0>();   // (ly_akv: the q group is the newest)\n  if constexpr (IDXF) {   // the q group", 1),
]
HEADER = ("# ly_akv: ly_v1un with the q_len 1 one-tile attention build's cached K/V copies issued before "
          "griddepcontrol.wait (tools/dl_ly_akv_patch.py).\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    start = 'ATT7_CUDA_SRC = r"""\n'
    i = text.index(start) + len(start)
    j = text.index('\n"""\n', i)
    at = text[i:j]
    for old, new, n in EDITS:
        at = sub(at, old, new, n, "att")
    text = HEADER + text[:i] + at + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
