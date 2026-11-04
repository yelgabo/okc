#!/usr/bin/env python3
"""ly_a16kv: the q_len 1 per-request multi-tile attention build (attn_kernel<1, false, 1>, decode 16 / 32 sliding
only) issues its first STAGES - 1 tiles' cached K/V before griddepcontrol.wait.

Base ly_stk2 (4bbf4010). KVPRE = L == 1 && !FLAT && MODE == 1 && !WS && !S2 is true for that one instance; every other
instance compiles the base code (if constexpr). In it:

- before the wait: the page-index group of tiles 0 .. STAGES - 2 plus the layer-input constants (inv_freq, k_scale,
  v_scale, cache_loc) as one cp.async group, waited (wait_group 0, CTA barrier), then the cached rows of tiles
  0 .. STAGES - 2 (rows at positions < seq; the new token's row is skipped and zero-fill past the range, as the
  base's prologue issue) with the next page indices, one group per tile as in the base;
- after the wait (unchanged position, trigger unchanged): q rows and the new k / v rows from qkv as direct
  ld.global.cg loads stored to shared memory instead of a cp.async group (so the q pass does not wait behind the
  K/V groups); the base's IDXF wait (wait_group STAGES - 1 + barrier) then only orders those stores.

Every operand read before the wait is a layer input that no kernel of the step writes (legality table:
docs/runs-decode-layer.md "Pre-wait legality per operand"); qkv, the cache row write and the workspace stay after it.
Group arithmetic in the tile loop is the base's (idx | tile 0 | tile 1 | in-loop tiles; the base had the q group
second). No fence, barrier, trigger or launch attribute moves.

    python3 tools/dl_ly_a16kv_patch.py [--base muse_decode_layer/cand/ly_stk2.py] [--out muse_decode_layer/cand/ly_a16kv.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk2.py"
DST = ROOT / "muse_decode_layer/cand/ly_a16kv.py"
PIN = "4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

E_IDXF = ("  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n",
          "  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n"
          "  // ly_a16kv: the q_len 1 per-request multi-tile build (decode 16 / 32 sliding) issues its first tiles' cached K/V\n"
          "  // before griddepcontrol.wait; page_table, seq_lens, the constants and the cached rows are layer inputs\n"
          "  constexpr bool KVPRE = L == 1 && !FLAT && MODE == 1 && !WS && !Cfg<L, FLAT>::S2;\n")

E_PRE = ("  cp_commit();\n  if (OKC_LATETID && !WS && (MODE & 8)) {",
         "  if constexpr (KVPRE) {\n"
         "    if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);\n"
         "    else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);\n"
         "    else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);\n"
         "    else if (tidL >= 16 && tidL < 16 + L) {\n"
         "      const int jj = tidL - 16;\n"
         "      asm volatile(\"cp.async.ca.shared.global [%0], [%1], 8;\\n\" ::\"r\"(smem_u32(&S.cloc[jj])), \"l\"(p.cache_loc + b * L + jj));\n"
         "    }\n"
         "  }\n"
         "  cp_commit();\n"
         "  if constexpr (KVPRE) {\n"
         "    cp_wait<0>();\n"
         "    __syncthreads();\n"
         "    DBG(1);\n"
         "#pragma unroll\n"
         "    for (int k = 0; k < STAGES - 1; ++k) {\n"
         "      if (k < ntiles) issue_kv(k, false);\n"
         "      issue_idx(k + STAGES - 1);\n"
         "      cp_commit();\n"
         "    }\n"
         "    DBG(8);\n"
         "  }\n"
         "  if (OKC_LATETID && !WS && (MODE & 8)) {")

Q_OLD = ("    cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);\n")
Q_NEW = ("    if constexpr (KVPRE)\n"
         "      *reinterpret_cast<uint4*>(&S.qst()[rr * 64 + ch * 8]) =\n"
         "          __ldcg(reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8));\n"
         "    else\n"
         "      cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);\n")
KV_OLD = "      cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);\n"
KV_NEW = ("      if constexpr (KVPRE)\n"
          "        *reinterpret_cast<uint4*>(&S.kvst[rr * 64 + ch * 8]) = __ldcg(reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col));\n"
          "      else\n"
          "        cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);\n")
C_OLD = ("  // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue\n"
         "  if (tidL < 8)")
C_NEW = ("  // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue\n"
         "  if constexpr (!KVPRE) {\n"
         "  if (tidL < 8)")
C2_OLD = ("    asm volatile(\"cp.async.ca.shared.global [%0], [%1], 8;\\n\" ::\"r\"(smem_u32(&S.cloc[jj])), \"l\"(p.cache_loc + b * L + jj));\n"
          "  }\n  cp_commit();\n")
C2_NEW = ("    asm volatile(\"cp.async.ca.shared.global [%0], [%1], 8;\\n\" ::\"r\"(smem_u32(&S.cloc[jj])), \"l\"(p.cache_loc + b * L + jj));\n"
          "  }\n  cp_commit();\n  }\n")
K_OLD = "  // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----\n"
K_NEW = "  if constexpr (!KVPRE) {\n" + K_OLD
K2_OLD = "  DBG(8);\n  }\n\n  if constexpr (IDXF) {"
K2_NEW = "  DBG(8);\n  }\n  }\n\n  if constexpr (IDXF) {"

EDITS = [E_IDXF, E_PRE, (Q_OLD, Q_NEW), (KV_OLD, KV_NEW), (C_OLD, C_NEW), (C2_OLD, C2_NEW), (K_OLD, K_NEW),
         (K2_OLD, K2_NEW)]
HEADER = ("# ly_a16kv: ly_stk2 with the decode 16 / 32 sliding attention build (attn_kernel<1, false, 1>) issuing its first "
          "tiles' cached K/V before griddepcontrol.wait, q / new k,v as direct loads after it (tools/dl_ly_a16kv_patch.py).\n")


def sub(text, old, new, what):
    if text.count(old) != 1:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    at = text[i:j]
    for n, (old, new) in enumerate(EDITS):
        at = sub(at, old, new, f"att edit {n}")
    text = HEADER + text[:i] + at + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
