#!/usr/bin/env python3
"""L1-bypassing loads for every read of a same-replay predecessor's output in the T <= 8 chain of the public
layer file (g1b4_stage_release.py, ba9e327c). Writes muse_decode_layer/cand/g1b4_pdl_ldcg.py.

The release launches no kernel as a PDL secondary (every programmatic attribute is set with
programmaticStreamSerializationAllowed = 0, Triton launch_pdl is False, the MoE is a cooperative launch), so these
reads are safe today. They become stale-L1 hazards the moment any of these edges is made a real PDL edge: a
dependent grid resident before its primary finishes can hit L1 lines holding the previous replay's values.

    python3 tools/dl_pdl_ldcg_patch.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "muse_decode_layer/cand/g1b4_stage_release.py"
DST = ROOT / "muse_decode_layer/cand/g1b4_pdl_ldcg.py"
PIN = "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"


def sub(s, old, new, count):
    n = s.count(old)
    if n != count:
        sys.exit(f"expected {count} of {old[:80]!r}, found {n}")
    return s.replace(old, new)


def scoped(s, start, end, edits):
    a = s.index(start)
    b = s.index(end, a)
    body = s[a:b]
    for old, new, count in edits:
        body = sub(body, old, new, count)
    return s[:a] + body + s[b:]


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("source is not the pinned public layer file")
    s = raw.decode()

    # attention: the output-gate rows are qkv, the qkv projection's output
    gate = "*reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)"
    s = sub(s, gate, "__ldcg(reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64"
                     " + (16 * pair + hh) * 64 + part * 8))", 4)

    # attention boundary b1 (both glue sources): the branch is the o projection's output
    s = sub(s, "      glb<V>(a.branch+row*8192+col,b+r*V);\n", "      glbcg<V>(a.branch+row*8192+col,b+r*V);\n", 2)
    glb_head = "template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {\n"
    glbcg = '''template<int V> __device__ __forceinline__ void glbcg(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.cg.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.cg.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
'''
    s = sub(s, glb_head, glbcg + glb_head, 2)

    # route: pooled and logits are the fused pooler + router's outputs
    s = scoped(s, "def _sl_tile_route(", "SL_TILE_CONFIG=", [
        ("tl.load(POOLED+t*4096+c)", "tl.load(POOLED+t*4096+c, cache_modifier='.cg')", 1),
        ("tl.load(ALL+t*4608+c)", "tl.load(ALL+t*4608+c, cache_modifier='.cg')", 1),
        ("tl.load(POOLED+t*4096+oc)", "tl.load(POOLED+t*4096+oc, cache_modifier='.cg')", 1),
        ("tl.load(ALL+t*4608+oc)", "tl.load(ALL+t*4608+oc, cache_modifier='.cg')", 1),
        ("tl.load(LOGITS + t * 256 + e)", "tl.load(LOGITS + t * 256 + e, cache_modifier='.cg')", 1),
        ("tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)",
         "tl.load(ALL + t * 4608 + 4096 + e, cache_modifier='.cg') + tl.load(ALL + t * 4608 + 4352 + e, cache_modifier='.cg')", 1),
    ])

    # finish: P is the unpooler's output, CARRY (s1) and TAIL the boundary's
    s = scoped(s, "def _sl_cluster_finish(", "SL_CLUSTER_CONFIG=", [
        ("tl.load(CARRY+t*8192+c)", "tl.load(CARRY+t*8192+c, cache_modifier='.cg')", 1),
        ("tl.load(TAIL+t)", "tl.load(TAIL+t, cache_modifier='.cg')", 1),
        ("tl.load(CARRY+t*8192+o)", "tl.load(CARRY+t*8192+o, cache_modifier='.cg')", 1),
        ("tl.load(P+t*1024+c)", "tl.load(P+t*1024+c, cache_modifier='.cg')", 1),
    ])

    # MoE (helper0 source, the Tiny / Tiny2 / MID builds): ids, weights and x are the route kernel's outputs
    s = scoped(s, "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"", "_MA_ALT_SRCS['moe_helper0df']", [
        ("__ldg(reinterpret_cast<const int4*>(p.topk_ids)", "__ldcg(reinterpret_cast<const int4*>(p.topk_ids)", 1),
        ("__ldg(p.topk_ids", "__ldcg(p.topk_ids", 5),
        ("__ldg(p.topk_w", "__ldcg(p.topk_w", 1),
        ("p.topk_w[tk * 8 + sl]", "__ldcg(p.topk_w + tk * 8 + sl)", 1),
        ("p.topk_w[tk * 8 + s]", "__ldcg(p.topk_w + tk * 8 + s)", 2),
        ("PB ? __ldcg(src + c * 8) : __ldg(src + c * 8)", "__ldcg(src + c * 8)", 1),
        ("PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1)", "__ldcg(src + c * 8 + 1)", 1),
        ("v = PB ? __ldcg(src) : __ldg(src);", "v = __ldcg(src);", 1),
        ("__ldg(reinterpret_cast<const uint4*>(p.x + ", "__ldcg(reinterpret_cast<const uint4*>(p.x + ", 1),
    ])

    s = s.replace("# okc: problem=muse_decode_layer\n",
                  "# okc: problem=muse_decode_layer\n# pdl_ldcg: L1-bypassing reads of same-replay predecessor outputs"
                  " in the T <= 8 chain (tools/dl_pdl_ldcg_patch.py on ba9e327c).\n", 1)
    DST.write_text(s)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
