"""Offline test of the M1 gate's census skip (user decision 2026-10-30): GATE_SRC and M1_SRC from
tools/dl_gate_patch.py run on CPU against a stub layer whose kernel_fn makes an M1 PDL launch
(counted as the runners count it) only at 8 tokens and above, as the real layer does with cgemv
o and the fp32 front at one token. An injected candidate-only difference in one replay (BF16
rounding level, as in d6c30f88c617438b; here relative about 2e-3) must be skipped at a shape
without an M1 launch, judged and rejected at a shape with one, and judged and rejected
wherever the census fails.

    uv run --with torch python tools/dl_m1_gate_census_test.py tools [LAYER.py]

With LAYER.py, the gate and M1 code come from that file instead (from the shared gate
comment to _moe1_tune and from the M1 negative-control comment to _m1_enable).
"""
import sys
import torch

sys.path.insert(0, sys.argv[1])
import dl_gate_patch as gp  # noqa: E402

torch.cuda.synchronize = lambda: None
ns = {"torch": torch, "_FAILED": [], "M1_BUCKETS": (8, 16, 32), "M1_STRESS_REPLAYS": 60,
      "HIDDEN": 256, "RMS_EPS": 1e-6}
if len(sys.argv) > 2:
    layer = open(sys.argv[2]).read()
    exec(layer[layer.index("# Fresh-input replay gate"):layer.index("def _moe1_tune(")], ns)
    if "\nM1_GATE_MODE = " in layer:
        exec(layer[layer.index("\nM1_GATE_MODE = "):].split("\n")[1], ns)
    a = layer.index("# The single-element negative control in the M1 gate")
    exec(layer[a:layer.index("def _m1_enable(state):", a)], ns)
    print(f"gate and M1 code from {sys.argv[2]}")
else:
    exec(gp.GATE_SRC, ns)
    exec(gp.M1_SRC, ns)
CFG = {"inject": None, "census_broken": False}
calls = {}


def m1_mode(tokens, state):
    if tokens > 32:
        return "off"
    key = next(b for b in (8, 16, 32) if b >= tokens)
    return state.get("m1_policy", {}).get(key, "off")


def kernel_fn(data, state):
    tokens = data["stream"].shape[0]
    mode = m1_mode(tokens, state)
    if mode != "off" and tokens >= 8 and ns["_M1_PDL_LAUNCHES"] is not None:
        ns["_M1_PDL_LAUNCHES"][0] += 2
    hid = (data["attn_input"].float() * 0.5).to(torch.bfloat16)
    stream = data["stream"] * 1.25
    if mode != "off" and CFG["inject"] == tokens:
        key = id(data)
        calls[key] = calls.get(key, 0) + 1
        if calls[key] == 20:
            hid = hid.float()
            hid[:, :32] *= 1.004
            hid = hid.to(torch.bfloat16)
    return hid, stream, torch.zeros((tokens, 8), dtype=torch.long)


class FakeGraph:
    def __init__(self, fn):
        self.fn = fn

    def replay(self):
        self.fn()


def graph_build(fn):
    for _ in range(3):
        fn()
    return FakeGraph(fn)


def timeline_case(state, batch, length, layer, seed):
    gen = torch.Generator().manual_seed(seed)
    stream = torch.randn((batch * length, ns["HIDDEN"]), generator=gen)
    return {"stream": stream, "attn_input": stream.to(torch.bfloat16)}


ns.update(kernel_fn=kernel_fn, _graph_build=graph_build, _timeline_case=timeline_case)


def run(inject, broken):
    CFG["inject"] = inject
    calls.clear()
    ns["_M1_PDL_LAUNCHES"] = None if broken else [0]
    try:
        return "pass", ns["_m1_stress"]({})
    except RuntimeError as exc:
        return "raise", [str(exc)]


cases = [
    ("clean", None, False, "pass", ("1x1s skipped: no M1 launch", "1x1f skipped: no M1 launch",
                                     "1x8s ok", "32x1f ok")),
    ("noise at 1 token, no M1 launch there", 1, False, "pass", ("1x1s skipped: no M1 launch",)),
    ("noise at 8 tokens, M1 launches there", 8, False, "raise", ("1x8s FAIL",)),
    ("census broken, noise at 1 token", 1, True, "raise", ("1x1s FAIL", "census failed")),
    ("census broken, clean", None, True, "pass", ("1x1s ok", "census failed")),
]
bad = 0
for name, inject, broken, want, needles in cases:
    got, lines = run(inject, broken)
    text = "\n".join(lines)
    ok = got == want and all(n in text for n in needles)
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}")
    for line in lines:
        print("     " + line[:150])
print("all cases as expected" if not bad else f"{bad} cases failed")
sys.exit(1 if bad else 0)
