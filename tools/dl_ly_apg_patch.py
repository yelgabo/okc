#!/usr/bin/env python3
"""ly_apg / ly_anpf: the pooler L2 prefetch the layer's T = 8 attention issues at entry, on ly_stk1.

Thread 0 of every attention CTA issues the pooler's first 4 MB (about 124 KB per CTA at 33 splits) as
cp.async.bulk.prefetch.L2 at kernel entry. Hypothesis from the round 2 stamps: the warp's later
fence.proxy.async.shared::cta (after the q pass: e4m3 q tile for the wgmma) waits for those async-proxy prefetches, so
thread 0's q pass reads 1.0 us while the same code run before griddepcontrol.wait takes no visible time (ly_aqw).

--mode off (ly_anpf, diagnostic): both attention launchers pass no prefetch at T = 8 (one Python condition each),
T = 1 unchanged.
--mode gen (ly_apg, candidate): at L == 8 the same byte range is prefetched with plain prefetch.global.L2 (generic
proxy, one 128-byte line per instruction), spread over the CTA's threads (4 lines per thread at 33 splits); every other
instance (q_len 1) keeps the bulk prefetch. Same source range, same issue point (kernel entry, before
griddepcontrol.wait; the pooler weights are prepare-time constants no layer kernel writes).

    python3 tools/dl_ly_apg_patch.py --mode gen [--src muse_decode_layer/cand/ly_stk1.py] [--out ...]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODE = sys.argv[sys.argv.index("--mode") + 1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / f"muse_decode_layer/cand/{'ly_apg' if MODE == 'gen' else 'ly_anpf'}.py"
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

BULK = '''  if (p.pf_chunk && threadIdx.x == 0) {
    const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;
    for (unsigned o = 0; o < p.pf_chunk; o += 32768u) {
      const unsigned n = min(32768u, p.pf_chunk - o);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + o), "r"(n) : "memory");
    }
  }
'''
GEN = '''  if (L == 8 && OKC_PFGEN8) {
    // (OKC_PFGEN8) generic-proxy line prefetches spread over the CTA: a bulk prefetch is an async-proxy operation the
    // issuing warp's fence.proxy.async after the q pass waited for (the whole CTA then waited at its next barrier)
    if (p.pf_chunk) {
      const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;
      for (unsigned o = threadIdx.x * 128u; o < p.pf_chunk; o += Cfg<L, FLAT>::NTX * 128u)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(src + o));
    }
  } else
''' + BULK
EDITS_GEN = [
    ("#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n",
     "#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n"
     "#define OKC_PFGEN8 1    // q_len 8: the pooler L2 prefetch as per-thread prefetch.global.L2 lines, not a bulk prefetch\n"),
    (BULK, GEN),
]
PY_OFF = ('if bst is not None and "pooler" in bst.get("sk2", {}) and cfg["batch"] * cfg["q_len"] in (1, 8):',
          'if bst is not None and "pooler" in bst.get("sk2", {}) and cfg["batch"] * cfg["q_len"] in (1,):')
HEADERS = {
    "gen": "# ly_apg: ly_stk1 + the T = 8 attention's pooler L2 prefetch as generic prefetch.global.L2 lines spread over "
           "the CTA instead of thread 0's cp.async.bulk.prefetch (tools/dl_ly_apg_patch.py --mode gen).\n",
    "off": "# ly_anpf: ly_stk1 with no pooler L2 prefetch from the T = 8 attention; diagnostic, not a candidate "
           "(tools/dl_ly_apg_patch.py --mode off).\n",
}


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    if MODE == "gen":
        i = text.index('ATT7_CUDA_SRC = r"""\n')
        j = text.index('\n"""\n', i)
        blk = text[i:j]
        for old, new in EDITS_GEN:
            blk = sub(blk, old, new, 1, "att7")
        text = text[:i] + blk + text[j:]
    elif MODE == "off":
        text = sub(text, PY_OFF[0], PY_OFF[1], 2, "launchers")
    else:
        sys.exit("--mode gen|off")
    text = HEADERS[MODE] + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
