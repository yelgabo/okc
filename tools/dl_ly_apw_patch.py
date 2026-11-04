#!/usr/bin/env python3
"""ly_apw: the layer attention's one-tile q_len 8 per-request builds (attn_kernel<8, false, MODE> with MODE & 4, the
verify 1 sliding instance <8, false, 15>) issue the cached window's K/V before griddepcontrol.wait.

Base ly_v1un (pinned). One change, in ATT7_CUDA_SRC's segment(), behind constexpr PREKV = OKC_PREKV8 && L == 8 && ONE
&& !WS && !S2 (every other instance compiles the base text):
- before pdl_wait: wait for the page-index group, CTA barrier, the OKC_SPEC length check, then the K/V copies of the
  CTA's tile (the ONE straight-line issue, unchanged text);
- after pdl_wait the q / new k,v / constants group is issued as before, then one cp.async.wait_group 0 + CTA barrier
  (the q group is the newest group now, so the base's two partial waits become one full wait).
Pre-wait operands: page_table, seq_lens (already pre-wait in the base), k_cache / v_cache rows at positions < seq.
Those are layer inputs; the only kernel of the layer that writes either cache is this one (the new rows at cache_loc,
after its wait), and the copies skip positions >= seq (zero fill past the range, patch from qkv after the wait).
Nothing global is written before the wait.

    python3 tools/dl_ly_apw_patch.py [--src F --out G]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_apw.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

SPEC_CHK = ("  if constexpr (!FLAT && (MODE & 8) != 0) {\n"
            "    // OKC_SPEC: this CTA's ranges and page indices came from the speculated length (see the kernel); the real\n"
            "    // cache_seqlens value has landed by now (it was loaded alongside the indices). On a miss, nothing global has been\n"
            "    // written: drain the copies and let the caller redo the segment with the real length (today's path)\n"
            "    if (seq_real_ != seq) {\n"
            "      cp_wait<0>();\n"
            "      __syncthreads();\n"
            "      return false;\n"
            "    }\n"
            "  }\n"
            "  DBG(1);\n")
KV_START = "  if constexpr (!WS) {\n    constexpr int PRE = Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1;"
KV_END = "  DBG(8);\n  }\n\n  if constexpr (IDXF) {"


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def patch_att(a):
    i = a.index(KV_START)
    j = a.index(KV_END, i)
    kv = a[i:j]   # the K/V issue block (straight-line ONE path and the general path), unchanged text
    a = sub(a, "#define OKC_PDLKV 0", "#define OKC_PREKV8 1    // q_len 8 one-tile per-request builds: the window's K/V issued before "
            "griddepcontrol.wait\n#define OKC_PDLKV 0", 1, "define")
    a = sub(a, "  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n",
            "  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n"
            "  constexpr bool PREKV = OKC_PREKV8 && L == 8 && ONE && !WS && !Cfg<L, FLAT>::S2;\n", 1, "prekv")
    pre = ("  if constexpr (PREKV) {   // page indices, cache_seqlens and cached K/V rows: layer inputs (see the q_len 8 notes)\n"
           "    cp_wait<0>();\n    csync<WS, NT>();\n" + SPEC_CHK.replace("\n  ", "\n    ").replace("  if constexpr", "    if constexpr", 1)
           + kv.replace("\n", "\n  ").rstrip(" ") + "    DBG(8);\n  }\n")
    a = sub(a, "\n\n  pdl_wait();   // qkv (q, new k / v, gate)", "\n" + pre + "\n  pdl_wait();   // qkv (q, new k / v, gate)", 1, "pre")
    old_post = "  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n" + SPEC_CHK + kv + "  DBG(8);\n  }\n"
    new_post = ("  if constexpr (PREKV) {\n    cp_wait<0>();\n    csync<WS, NT>();\n  } else {\n"
                + old_post[:-len("  DBG(8);\n  }\n")] + "  DBG(8);\n  }\n  }\n")
    a = sub(a, old_post, new_post, 1, "post")
    a = sub(a, "  if constexpr (IDXF) {   // the q group", "  if constexpr (IDXF && !PREKV) {   // the q group", 1, "idxf")
    return a


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    start = 'ATT7_CUDA_SRC = r"""\n'
    i = text.index(start) + len(start)
    j = text.index('\n"""\n', i)
    text = text[:i] + patch_att(text[i:j]) + text[j:]
    text = ("# ly_apw: ly_v1un + the q_len 8 one-tile attention builds issue the window's K/V before griddepcontrol.wait "
            "(tools/dl_ly_apw_patch.py).\n") + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
