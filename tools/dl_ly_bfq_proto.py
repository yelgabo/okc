"""Box-only prototype: muse_decode_layer/cand/ly_stk3.py with qkv at T = 128 on the BF16 board's tuned own kernel
(muse_bf16_proj/submission.py loaded as a module, so the file is not a valid submission) instead of the cuBLASLt
plan. It measures the swap's value inside the layer before porting the kernel. `state["bfq_on"]` switches it per
arm (tools/ly_arms.py).

    python3 tools/dl_ly_bfq_proto.py   # writes muse_decode_layer/cand/ly_bfq_proto.py
"""
import hashlib
import pathlib
import py_compile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
DST = ROOT / "muse_decode_layer/cand/ly_bfq_proto.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"

TAIL = r'''


# ---- bfq proto (tools/dl_ly_bfq_proto.py, box only) ----
import importlib.util as _bfq_iu

_BFQ = {"mod": None, "st": None, "plan": None, "on": False}


def _bfq_load(weights):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "muse_bf16_proj", "submission.py")
    spec = _bfq_iu.spec_from_file_location("bfq_bf16_sub", os.path.abspath(p))
    m = _bfq_iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    w = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler")}
    w["router"] = weights["router"].float().contiguous()
    return m, m.prepare(w)


_bfq_prior_lt_run = _graft_lt_run
def _graft_lt_run(x, plan):
    if _BFQ["on"] and plan is _BFQ["plan"]:
        n, k = _BFQ["nk"]
        return _BFQ["mod"].kernel_fn({"x": x, "config": {"layer": "qkv_gate", "m": x.shape[0], "n": n, "k": k}},
                                     _BFQ["st"])
    return _bfq_prior_lt_run(x, plan)


_bfq_prior_kernel = kernel_fn
def kernel_fn(data, state):
    _BFQ["on"] = bool(state.get("bfq_on", False))
    try:
        return _bfq_prior_kernel(data, state)
    finally:
        _BFQ["on"] = False


_bfq_prior_prepare = prepare
def prepare(weights):
    st = _bfq_prior_prepare(weights)
    _BFQ["mod"], _BFQ["st"] = _bfq_load(weights)
    _BFQ["plan"] = st.get("graft_lt", {}).get(("qkv", 128))
    _BFQ["nk"] = tuple(weights["qkv_gate"].shape)
    pk = _BFQ["st"].get("pick", {}).get(("qkv_gate", 128))
    print(f"BFQ proto: qkv 128 plan {'found' if _BFQ['plan'] is not None else 'MISSING'}, bf16 pick {pk}"[:300],
          flush=True)
    st["bfq_on"] = True
    return st
'''


def main():
    src = SRC.read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == PIN, "base changed"
    assert "\nimport os\n" in src
    DST.write_text(src + TAIL)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
