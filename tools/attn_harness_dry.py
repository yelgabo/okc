"""CPU dry run of bench.py's self-test harness: the input builder, the FP32 reference and the
judge-style check, on small shapes, with a 'kernel' that writes the reference's own new rows
and returns the reference output (must PASS), and a perturbed output (must FAIL)."""
import ast
import sys

import torch

src = open(sys.argv[1]).read()
tree = ast.parse(src)
ns = {"torch": torch, "RMS_EPS": 1e-5, "FP8": torch.float8_e4m3fn}
want = {"_OB_HEADS", "_OB_WINDOW", "_OB_THETA", "_ob_rms", "_ob_rope", "_ob_inputs",
        "_ob_reference", "_ob_rel", "_ob_check"}
for node in tree.body:
    names = set()
    if isinstance(node, ast.Assign):
        names = {getattr(t, "id", "") for t in node.targets}
    elif isinstance(node, ast.FunctionDef):
        names = {node.name}
    if names & want:
        code = ast.get_source_segment(src, node).replace('dev = "cuda"', 'dev = "cpu"')
        exec(code, ns)

shapes = [
    (2, 100, "tp8", "sliding", 8, "fixed", "edges", 1, 1.0),
    (3, 70, "tp8", "sliding", 1, "mixed", "soft", 64, 0.015625),
    (2, 150, "tp8", "full", 8, "fixed", "coherent", 1, 1.0),
    (2, 90, "tp8", "full", 1, "mixed", "normal", 64, 1.0),
    (1, 2100, "tp8", "sliding", 8, "fixed", "edges", 1, 1.0),
]
for shp in shapes:
    data = ns["_ob_inputs"](*shp)
    cfg = data["config"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    q_in, k_in, v_in, gate = data["qkv"].split([hq * 64, hkv * 64, hkv * 64, hq * 64], -1)
    pos = data["seq_lens"].long()[:, None] + torch.arange(cfg["q_len"])
    k = ns["_ob_rms"](k_in.reshape(cfg["batch"], cfg["q_len"], hkv, 64))
    if cfg["rope_theta"]:
        k = ns["_ob_rope"](k, pos)
    k_new = (k / data["k_scale"]).to(torch.float8_e4m3fn)
    v_new = (v_in.reshape(cfg["batch"], cfg["q_len"], hkv, 64).float() / data["v_scale"]).to(
        torch.float8_e4m3fn)
    loc = data["cache_loc"]
    data["k_cache"].view(-1, hkv, 64)[loc] = k_new.reshape(-1, hkv, 64)
    data["v_cache"].view(-1, hkv, 64)[loc] = v_new.reshape(-1, hkv, 64)
    out = ns["_ob_reference"](data, False, (k_new, v_new)).to(torch.bfloat16)
    ok, msg = ns["_ob_check"](data, out)
    bad = (out.float() * 1.3).to(torch.bfloat16)
    ok2, msg2 = ns["_ob_check"](data, bad)
    print(shp[:5], shp[6:], "exact:", "PASS" if ok else "FAIL", msg, "| perturbed:",
          "FAIL" if not ok2 else "PASS?!", msg2.split(" worst")[0])
