#!/usr/bin/env python3
"""ly_apf: the attention's L2 prefetch of the pooler stream extended to decode 16 / 32 (T = 16 / 32).

Base: muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a1..., checked). At T = 1 / 8 the attention launcher passes
(pooler stream, 4 MB) and every attention CTA's thread 0 issues `cp.async.bulk.prefetch.L2` over its share of the
first 4 MB of the pooler's 12-bit stage-major stream (attn_kernel entry, `p.pf_chunk`), so the pooler's first stages
read L2. At T = 16 / 32 the gate `cfg["batch"] * cfg["q_len"] in (1, 8)` leaves it off, although the pooler there
(the FUSE sk2, pooler + router in one launch) reads the same `state["bfp"]["sk2"]["pooler"]` stream: FUSE is
static_asserted to the pooler config's geometry, so the layout is the T = 1 / 8 one. The attention at decode 16 / 32
sliding is latency-bound (7.7 / 9.7 us of critical path for 8.4 / 16.8 MB of KV), so DRAM has room beside it.

Change: the gate and the 4 MB cap become a table, `APF_BYTES = {1: 4 MB, 8: 4 MB, 16: A, 32: B}` (T absent: no
prefetch), in both copies of the launcher wrapper (the base `_decode_attention7` and `_pdl8_decode_attention7`).
T = 1 / 8 keep 4 MB. The prefetch writes nothing and orders nothing; the kernel, its wait and its trigger are the
base's (the prefetch already runs at entry at T = 1 / 8).

    python3 tools/dl_ly_apf_patch.py [--apf 16:8,32:8] (MB) [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_apf.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
APF = {1: 4, 8: 4, 16: 8, 32: 8}
if "--apf" in sys.argv:
    APF.update({int(k): int(v) for k, v in (kv.split(":") for kv in sys.argv[sys.argv.index("--apf") + 1].split(","))})

EDITS = [
    ('and "pooler" in bst.get("sk2", {}) and cfg["batch"] * cfg["q_len"] in (1, 8):',
     'and "pooler" in bst.get("sk2", {}) and APF_BYTES.get(cfg["batch"] * cfg["q_len"], 0) > 0:', 2),
    ('        prefetch = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))',
     '        prefetch = (_ps.data_ptr(), min(APF_BYTES[cfg["batch"] * cfg["q_len"]], _ps.numel() * _ps.element_size()))', 2),
]


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new, n in EDITS:
        if text.count(old) != n:
            sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old[:80]!r}")
        text = text.replace(old, new)
    table = "{" + ", ".join(f"{t}: {mb} << 20" for t, mb in sorted(APF.items())) + "}"
    text = (f"# {DST.stem}: ly_stk1 with the attention's pooler L2 prefetch at T = 16 / 32 too, MB by T {APF} "
            f"(tools/dl_ly_apf_patch.py).\n" + text +
            f"\n\n# apf (tools/dl_ly_apf_patch.py): bytes of the pooler stream the attention prefetches into L2, by T\n"
            f"APF_BYTES = {table}\n")
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
