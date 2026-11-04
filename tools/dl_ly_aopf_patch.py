#!/usr/bin/env python3
"""ly_aopf: the decode 16 / 32 sliding attention (attn_kernel<1, false, 1>) pulls the o projection's sk2 weight stream
into L2 during its own merge tail, when its K/V reads are done and DRAM is otherwise idle until o's
griddepcontrol.wait returns.

Base ly_stk3 (a1c48810). Round 1 stamps on the same device code (st_a16o2.log): the tile loop ends 4.77 / 7.22 us
after qkv's end, o's wait returns 8.43 / 10.13 and o then streams its 12-bit codes from DRAM for 3.27 / 4.16 us.
ly_apf (ruled out) prefetched the pooler from the attention's entry, during the K/V stream; the ruled-out row for the
side-stream prefetch names this form (o weights only, where HBM is idle) as untried.

The kernel's existing L2 prefetch (p.pf / p.pf_chunk, entry, pooler at T = 1 / 8) is moved for that one instance:
--at ticket (OKC_AOPF 1): thread 0 of every CTA that is not its request's last, right after its ticket returns;
--at loop (OKC_AOPF 2): thread 0 of every CTA, right after its tile loop, before the merge reds.
`cp.async.bulk.prefetch.L2` writes nothing and orders nothing; no fence, barrier, acquire, release, atomic, wait,
trigger or launch attribute is added, removed or moved. Host: both decode-attention wrappers pass o's sk2 stream
(prepare-time weights, a layer input) as the prefetch at q_len 1, batch 16 / 32, sliding only; every other launch
keeps its prefetch argument, and every other instance compiles the base code.

    python3 tools/dl_ly_aopf_patch.py --at ticket|loop [--base F] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"
AT = sys.argv[sys.argv.index("--at") + 1] if "--at" in sys.argv else "ticket"
MODE = {"ticket": 1, "loop": 2}[AT]
DST = ROOT / f"muse_decode_layer/cand/{'ly_aopf' if AT == 'ticket' else 'ly_aopfl'}.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

GATE = "(OKC_AOPF && L == 1 && !FLAT && MODE == 1)"
ATT_EDITS = [
    ("#define OKC_IDXFIRST 1 ",
     f"#define OKC_AOPF {MODE}   // ly_aopf: decode 16 / 32 sliding pulls o's weight stream into L2 in its merge tail "
     "(1 after a non-last CTA's ticket, 2 after the tile loop)\n#define OKC_IDXFIRST 1 "),
    ("// One (request b, KV pair) segment [start, end) of the request's key positions.\n",
     "__device__ __forceinline__ void aopf_issue(const Params& p) {\n"
     "  const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;\n"
     "  for (unsigned o = 0; o < p.pf_chunk; o += 32768u)\n"
     "    asm volatile(\"cp.async.bulk.prefetch.L2.global [%0], %1;\" :: \"l\"(src + o), \"r\"(min(32768u, p.pf_chunk - o)) : \"memory\");\n"
     "}\n\n"
     "// One (request b, KV pair) segment [start, end) of the request's key positions.\n"),
    ("  if (p.pf_chunk && threadIdx.x == 0) {\n",
     f"  if (p.pf_chunk && threadIdx.x == 0 && !{GATE}) {{\n"),
    ("    S.flag = (old + mine == n_keys);\n",
     "    S.flag = (old + mine == n_keys);\n"
     f"    if constexpr (OKC_AOPF == 1 && L == 1 && !FLAT && MODE == 1) {{ if (p.pf_chunk && old + mine != n_keys) aopf_issue(p); }}\n"),
    ("  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n",
     "  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n"
     "  if constexpr (OKC_AOPF == 2 && L == 1 && !FLAT && MODE == 1) { if (tid == 0 && p.pf_chunk) aopf_issue(p); }\n"),
]
PY_OLD = ('    if bst is not None and "pooler" in bst.get("sk2", {}) and cfg["batch"] * cfg["q_len"] in (1, 8):\n'
          '        _ps = bst["sk2"]["pooler"][1]\n'
          '        prefetch = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))\n')
PY_NEW = PY_OLD + (
    '    elif bst is not None and cfg["window"] and cfg["q_len"] == 1 and cfg["batch"] in (16, 32):\n'
    '        _os = bst.get("sk2", {}).get(bfp__sk2_name("o", cfg["batch"]))\n'
    '        if _os is not None:\n'
    '            prefetch = _aopf_view(_os[1], cfg["batch"])\n')
TAIL = '''

_AOPF_SEEN = set()


def _aopf_view(stream, T):
    nb = stream.numel() * stream.element_size()
    if T not in _AOPF_SEEN:
        _AOPF_SEEN.add(T)
        print(f"AOPF T={T} o stream {bfp__sk2_name('o', T)} {nb} B prefetched by the attention's merge tail", flush=True)
    return (stream.data_ptr(), nb)
'''
HEADER = (f"# {DST.stem}: ly_stk3 with the decode 16 / 32 sliding attention prefetching o's weight stream into L2 "
          f"{'after each non-last CTA ticket' if MODE == 1 else 'after each CTA tile loop'} (tools/dl_ly_aopf_patch.py).\n")


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
    at = text[i:j]
    for n, (old, new) in enumerate(ATT_EDITS):
        at = sub(at, old, new, 1, f"att edit {n}")
    text = text[:i] + at + text[j:]
    text = sub(text, PY_OLD, PY_NEW, 2, "py prefetch")
    text = HEADER + text + TAIL
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
