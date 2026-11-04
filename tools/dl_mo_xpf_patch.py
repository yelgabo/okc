#!/usr/bin/env python3
"""L2 prefetch one unit past the ring in the layer's skinny WS builds (TINY2 at verify 1, MID at decode 16).

    python3 tools/dl_mo_xpf_patch.py            # muse_decode_layer/cand/mo_xpf.py: only the next job's first unit
    python3 tools/dl_mo_xpf_patch.py --every    # muse_decode_layer/cand/mo_xpf2.py: every unit, one past the ring

Base: rv_k_combo2n.py (3ac83fb4, slack-free). The T = 8 timeline ("T = 8 MoE floor and gap", docs/runs-moe.md)
puts about 2.5 us of phase A's excess in weight waits that fall from about 1 us to 0.3 us when the weights are
already in L2. The ring streams a flat per-CTA sequence of units (A, then B) OKC_NS = 3 ahead, and its
in-flight bytes (132 CTAs x 3 x 36,864 B = 14.6 MB) barely cover HBM latency times bandwidth. A fourth slot
lost (+2 us at verify 1, the knob sweep), so this adds depth without a slot. When thread 0 issues sequence index
q + NS into the ring, it also issues `cp.async.bulk.prefetch.L2` for index q + NS + 1. With OKC_L2PF=1 that
happens only when that unit opens a new job (the brief: the next job's first refill early); with OKC_L2PF=2 it
happens for every unit. The prefetch only warms L2: no smem, no mbarrier, no write, so every fence, barrier, the
ring discipline and the final pass are unchanged. B4 / MAIN (decode 32 and up) are a different source and keep
their code.
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAND = ROOT / "muse_decode_layer/cand"
SRC, PIN = "rv_k_combo2n.py", "3ac83fb437d05308c2871301a5a10c537d9b2ec504d55305403641beec2f5d27"
MARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
TINY2_ADD = 'MOE_TINY2_DEFINES += ("@moe_helper0",)\n'

FUNCS = (
    "#ifndef OKC_L2PF\n#define OKC_L2PF 0\n#endif\n"
    "#if OKC_L2PF\n"
    "// sequence index q of this CTA (A units, then B units): the unit's bytes in the expert bank, and a job key\n"
    "__device__ __forceinline__ const unsigned char* ws_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0, int& key) {\n"
    "  const bool PB = q >= nA;\n"
    "  const int v = PB ? b0 + (q - nA) : a0 + q;\n"
    "  const int rbs = PB ? RB_B : RB_A;\n"
    "  const int j = S.njF + v / rbs, rb = v - (v / rbs) * rbs;\n"
    "  key = PB ? j + JMAX : j;\n"
    "  return (PB ? p.wBu : p.wAu) + ((size_t)S.jE[j] * rbs + rb) * (size_t)UNIT_BYTES;\n"
    "}\n"
    "// thread 0, after issuing index q0 into the ring: warm L2 with index q0 + 1 (L2PF 1: only when it opens a new job)\n"
    "__device__ __forceinline__ void ws_pf(const Params& p, const Smem& S, int q0, int nA, int a0, int b0) {\n"
    "  int k0, k1;\n"
    "  (void)ws_src(p, S, q0, nA, a0, b0, k0);\n"
    "  const unsigned char* src = ws_src(p, S, q0 + 1, nA, a0, b0, k1);\n"
    "  if (OKC_L2PF == 2 || k1 != k0)\n"
    "    asm volatile(\"cp.async.bulk.prefetch.L2.global [%0], %1;\\n\" ::\"l\"(src), \"r\"(UNIT_BYTES) : \"memory\");\n"
    "}\n"
    "#endif\n"
)
DECL = "__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\n"
REFILL = "      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n"
REFILL_NEW = REFILL + (
    "#if OKC_L2PF\n"
    "      if (q + OKC_NS + 1 < N) ws_pf(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2]);\n"
    "#endif\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", action="store_true")
    a = ap.parse_args()
    mode, dst_name = (2, "mo_xpf2.py") if a.every else (1, "mo_xpf.py")
    raw = (CAND / SRC).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit(f"{SRC} is not {PIN[:8]}")
    whole = raw.decode()
    assert whole.count(MARK) == 1
    head, rest = whole.split(MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    for old, new in ((DECL, DECL + FUNCS), (REFILL, REFILL_NEW)):
        if helper.count(old) != 1:
            sys.exit(f"helper0 anchor found {helper.count(old)} times: {old[:80]!r}")
        helper = helper.replace(old, new)
    if tail.count(TINY2_ADD) != 1:
        sys.exit("TINY2 define anchor")
    tail = tail.replace(TINY2_ADD, TINY2_ADD + f'MOE_TINY2_DEFINES += ("-DOKC_L2PF={mode}",)\n'
                        f'MOE_MID_DEFINES += ("-DOKC_L2PF={mode}",)\n')
    out = head + MARK + helper + tail
    dst = CAND / dst_name
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
