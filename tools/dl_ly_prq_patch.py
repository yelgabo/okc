#!/usr/bin/env python3
"""ly_prq: the router's four K-quarter partials are summed by the route kernel, not by a last-arriving pooler CTA.

Base: muse_decode_layer/cand/ly_micro2.py (SHA256 0bc2bd8b..., checked). Target: the tail of the fused pooler + router
kernel (docs/runs-decode-layer.md, "Pooler + router timeline"). Each of the 128 CTAs stores its 8-row x K/4 router
partial, then tid 0 does atom.add.release.gpu on its tile's counter; the CTA that brings a tile's count to 4 issues
fence.acq_rel.gpu, reads the four partials back and writes the logits. On the last CTA that counter round trip and
read-back cost about 1.2 us between its main loop and its pooler store.

Change. The fused kernel skips the counter block when it is launched with outr == nullptr (the launchers pass nullptr
for an empty outr tensor); the partial stores, the barrier after them and the pooler epilogue are unchanged. The
fused-launch Python wrappers (_gle_frt at T <= 8, bfp_fused_pooler_router at T = 16 / 32) pass an empty outr and
return the partial buffer, tagged `_prq_part`, as the logits. _sl_tile_route_cg (the route kernel the layer launches
after the fused kernel at T = 1, 8, 16, 32) is wrapped: given a tagged buffer it launches _prq_route_cg, a copy whose
SPLIT logit load becomes ((p0 + p1) + p2) + p3 of the four quarter partials, the same FP32 adds in the same order as
the last-arriving CTA did, so the logits are bitwise unchanged. Every other kernel that takes logits (_route,
_graft_route, _sl_graft_route, _sl_tile_route, _gle_route) is wrapped to raise if it ever receives a tagged buffer.
Ordering: the route kernel's loads stay after its griddepcontrol.wait (the fused kernel complete and flushed); no
fence, barrier or wait is removed from any kernel that still runs; the counter is no longer touched.

    python3 tools/dl_ly_prq_patch.py [--out muse_decode_layer/cand/ly_prq.py]
"""
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dl_ly_prtl_patch as tlp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_micro2.py"
DST = ROOT / "muse_decode_layer/cand/ly_prq.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd"

K_EDITS = [
    ("    __shared__ int s_rlast;\n    if (tid == 0) {\n      int old;\n",
     "    __shared__ int s_rlast;\n"
     "    if (outr != nullptr) {   // prq: outr == nullptr leaves the quarter sum to the route kernel (partials above)\n"
     "    if (tid == 0) {\n      int old;\n"),
    ("    named_bar(1, NC);   // the ring (redr) is reused by the pooler epilogue below\n",
     "    named_bar(1, NC);   // the ring (redr) is reused by the pooler epilogue below\n    }\n"),
]
F_EDITS = [
    ("outr.data_ptr<float>(), part.data_ptr<float>(), ",
     "(outr.numel() ? outr.data_ptr<float>() : nullptr), part.data_ptr<float>(), ", 2),
]

ROUTE_DEF = "@triton.jit\ndef _sl_tile_route_cg("
OLD_LOGIT = "            logit = tl.load(LOGITS + t * 256 + e, cache_modifier='.cg')\n"
NEW_LOGIT = ("            # prq: LOGITS is the fused kernel's quarter-partial buffer [32 tiles][4 quarters][32 tokens][8 rows];\n"
             "            # the four quarters are added in quarter order, as the last-arriving CTA did\n"
             "            pq = LOGITS + (e >> 3) * 1024 + t * 8 + (e & 7)\n"
             "            logit = (((tl.load(pq, cache_modifier='.cg') + tl.load(pq + 256, cache_modifier='.cg'))\n"
             "                      + tl.load(pq + 512, cache_modifier='.cg')) + tl.load(pq + 768, cache_modifier='.cg'))\n")

TAIL = r'''


# ---- prq (tools/dl_ly_prq_patch.py): router quarter sum in the route kernel ----
def _prq_tagged(m):
    part = m["fused_part"].view(-1)
    part._prq_part = True
    return part


def _gle_frt(xb, x32, state):
    """Fused BF16 pooler + FP32 router for M = xb.shape[0] <= 8 rows: (pooled bf16 [M, 4096], the router's tagged
    quarter partials; the route kernel adds them)."""
    bst = state["bfp"]
    m = xb.shape[0]
    dev = xb.device
    if "fused_part" not in bst:
        bst["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
        bst["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
    pci, pstream, ptab_, ppmax, pb4, _ = bst["sk2"]["pooler"]
    pooled = torch.empty((m, 4096), device=dev, dtype=torch.bfloat16)
    nolog = torch.empty((0,), device=dev, dtype=torch.float32)
    _bfp_pdl(m, "pool")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_pr(pstream, bst["sk2"]["router"][1], xb, pooled, nolog, ptab_, ppmax, pb4, bst["fused_part"], bst["fused_cnt"],
               m, x32, state["router_fp32"], int(GLE_ILV))
    return pooled, _prq_tagged(bst)


def bfp_fused_pooler_router(x, state):
    """Pooler and router in one launch (m <= 32, sm_90): (pooler [m, 4096] bf16, the router's tagged quarter partials)."""
    m = x.shape[0]
    assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]
    ci, stream, tab, pmax, b4, n_rows = state["sk2"]["pooler"]
    rstream = state["sk2"]["router"][1]
    if "fused_part" not in state:
        state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
        state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    nolog = torch.empty((0,), dtype=torch.float32, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2f(stream, rstream, x, out, nolog, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, bfp__sk2_tokr(m))
    return out, _prq_tagged(state)


class _PrqLaunch:
    def __init__(self, base, alt, name):
        self.base, self.alt, self.name = base, alt, name

    def __getitem__(self, grid):
        def go(*a, **kw):
            if any(getattr(x, "_prq_part", False) for x in a[:3]):
                if self.alt is None or not kw.get("SPLIT", False):
                    raise RuntimeError(f"prq: {self.name} got the router's quarter partials")
                return self.alt[grid](*a, **kw)
            return self.base[grid](*a, **kw)
        return go


_sl_tile_route_cg = _PrqLaunch(_sl_tile_route_cg, _prq_route_cg, "_sl_tile_route_cg")
_sl_tile_route = _PrqLaunch(_sl_tile_route, None, "_sl_tile_route")
_route = _PrqLaunch(_route, None, "_route")
_graft_route = _PrqLaunch(_graft_route, None, "_graft_route")
_sl_graft_route = _PrqLaunch(_sl_graft_route, None, "_sl_graft_route")
_gle_route = _PrqLaunch(_gle_route, None, "_gle_route")
'''


def build(whole):
    out = tlp.patch(whole, K_EDITS, F_EDITS)
    i = out.index(ROUTE_DEF)
    j = out.index("\n\n\n@triton.jit", i)
    fn = out[i:j]
    assert fn.count(OLD_LOGIT) == 1
    alt = fn.replace("def _sl_tile_route_cg(", "def _prq_route_cg(").replace(OLD_LOGIT, NEW_LOGIT)
    out = out[:j] + "\n\n\n" + alt + out[j:]
    return out.rstrip("\n") + TAIL


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    out = build(raw.decode())
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
