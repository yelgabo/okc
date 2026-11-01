"""Build a private G verify-32 sliding arm from the scored stage source.

Keep the selected native device literals and callable bodies. Replace the broad
preparation and fallback dispatch with explicit selected-load setup. This arm
cannot serve as a public submission.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PIN = "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"

CUSTOM = '''
_GRAFT_LT_T = (256,)
_MA_ALT_SRCS = {}

def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state = {
        "qkv": weights["qkv_gate"].contiguous(),
        "o": weights["o"].contiguous(),
        "pr": torch.cat([weights["pooler"], hi, lo], 0).contiguous(),
        "unpool": weights["unpooler"].contiguous(),
        "bias": weights["router_bias"].float().contiguous(),
        "g_preexp": weights["pre_expert_norm"].contiguous(),
        "attention_scratch": {"rows": {}, "cluster": {}, "retained": []},
        "ma_ext": {},
    }
    graft = {}
    for i, name in ((1, "attn_residual_gate"), (2, "ffn_residual_gate")):
        gate = weights[name].float() / 0.3
        beta = torch.sigmoid(gate)
        graft["alpha%d" % i] = torch.sqrt(torch.clamp(
            torch.sigmoid(-gate) * (1.0 + beta), min=1e-3)).contiguous()
        graft["beta%d" % i] = beta.contiguous()
    for key, name in (("pre1", "pre_ffn_norm"), ("post2", "post_ffn_norm"),
                      ("pre2", "next_pre_attn_norm")):
        graft[key] = (weights[name].float() + 1).to(torch.bfloat16).contiguous()
    state["graft"] = graft
    state["ma_sfx"] = _MA_SFX_IF_PRMT if _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"]) else ()
    state["ma_sc6"] = bool(SC6_WGS and state["ma_sfx"] and _wgs_on()
                           and _sc6_ok(weights["w13_sf"], weights["w2_sf"]))
    if not state["ma_sc6"]:
        raise RuntimeError("private G arm requires the scored native SC6 route")
    _fork_native_ma_prepare(weights, state)
    state["nsm"] = torch.cuda.get_device_properties(device).multi_processor_count
    flags = tuple(_main_defines(state["ma_sfx"])) + ("-DOKC_SC6W=1",) + tuple(state["ma_sfx"])
    if any(flag.startswith("@") for flag in flags):
        raise RuntimeError("private G arm selected an alternate source")
    state["native"] = _RouterExt(_ma_ext(flags))
    state["native"].okc_set_pdl(0)
    _att7()
    _graft_lt_plan(state)
    if set(state["graft_lt"]) != {(name, 256) for name in ("qkv", "o", "pr", "un")}:
        raise RuntimeError("private G arm Lt plan drift")
    torch.cuda.synchronize()
    _check_guards()
    return state

def kernel_fn(data, state):
    config = data["config"]
    expected = {"batch": 32, "q_len": 8, "tokens": 256, "window": 2048,
                "page_size": 1, "hidden": 8192, "num_q_heads": 16,
                "num_kv_heads": 2, "head_dim": 64, "top_k": 8,
                "n_experts": 256, "local_experts": 32, "kv_scale": 1.0}
    if any(config.get(name) != value for name, value in expected.items()):
        raise RuntimeError("private G arm supports verify32 sliding/page1 only")
    if data["page_table"].shape != (32, 86008):
        raise RuntimeError("private G arm page capacity drift")
    t = 256
    device = data["attn_input"].device
    plans = state["graft_lt"]
    qkv = _graft_lt_run(data["attn_input"], plans["qkv", t])
    attention = _decode_attention7(data, state, qkv)
    branch = _graft_lt_run(attention, plans["o", t])
    stream1 = torch.empty((t, 8192), device=device, dtype=torch.float32)
    hidden = torch.empty((t, 8192), device=device, dtype=torch.bfloat16)
    g = state["graft"]
    _graft_row_boundary[(t,)](branch, data["stream"], g["post2"], g["pre1"],
        g["alpha1"], g["beta1"], hidden, stream1, 8192, 8192, False, 4, 16,
        num_warps=16, launch_pdl=False)
    combined = _graft_lt_run(hidden, plans["pr", t])
    ids = torch.empty((t, 8), device=device, dtype=torch.int32)
    lids = torch.empty_like(ids)
    scores = torch.empty((t, 8), device=device, dtype=torch.float32)
    expert_input = torch.empty((t, 4096), device=device, dtype=torch.bfloat16)
    _graft_route[(t, 2)](combined, combined, combined, state["bias"], state["g_preexp"],
        ids, scores, expert_input, data["local_start"], lids,
        SPLIT=False, EARLY=_GRAFT_ROUTE_EARLY, PACKED=1, num_warps=4, launch_pdl=False)
    expert_output = torch.empty_like(expert_input)
    state["native"].okc_run(state["ma_w"], state["ma_scratch"], expert_input, lids,
        scores, expert_output, 0, 32, state["nsm"])
    branch = _graft_lt_run(expert_output, plans["un", t])
    stream2 = torch.empty_like(stream1)
    output = torch.empty_like(hidden)
    _graft_row_boundary[(t,)](branch, stream1, g["post2"], g["pre2"],
        g["alpha2"], g["beta2"], output, stream2, 8192, 8192, True, 4, 16,
        num_warps=16, launch_pdl=False)
    return output, stream2, ids
'''


def build(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PIN
    # This public file is an unchanged pure prefix plus startup-only validation.
    source = source[:861924]
    assert hashlib.sha256(source.encode()).hexdigest() == "295ca590a9d07a2491d600e1c96ec5583d29ae3220f619bc9e99adf62ef48491"
    tree = ast.parse(source)
    overrides = {n.name for n in ast.parse(CUSTOM).body if isinstance(n, ast.FunctionDef)}
    overrides.update(("_GRAFT_LT_T", "_MA_ALT_SRCS"))
    bindings = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [n.id for target in node.targets for n in ast.walk(target)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        else:
            continue
        for name in names:
            if name not in overrides:
                bindings[name] = node
    pending = {n.id for n in ast.walk(ast.parse(CUSTOM))
               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    seen, kept = set(), set()
    while pending - seen:
        name = (pending - seen).pop()
        seen.add(name)
        if name in bindings:
            node = bindings[name]
            kept.add(id(node))
            pending.update(n.id for n in ast.walk(node)
                           if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    lines = source.splitlines(keepends=True)
    selected = [n for n in tree.body if id(n) in kept or isinstance(n, (ast.Import, ast.ImportFrom))]
    bodies = []
    for node in selected:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        bodies.append("".join(lines[start - 1:node.end_lineno]))
    notices = "".join(lines[tree.body[-1].end_lineno:])
    assert not ast.parse(notices).body
    output = "# okc: problem=muse_decode_layer\n" + "\n\n".join(bodies) + CUSTOM + notices
    ast.parse(output)
    manifest = {
        "source_pin": PIN,
        "bytes": len(output.encode()),
        "sha256": hashlib.sha256(output.encode()).hexdigest(),
        "selected_native_literals": {n.targets[0].id: len(n.value.value.encode()) for n in selected
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            and len(n.value.value) > 10000},
        "source_review_and_host_tests_required": True,
        "gpu_execution": False,
    }
    return output, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out, manifest = build(args.source.read_text(encoding="utf8"))
    args.output.write_bytes(out.encode())
    print(json.dumps(manifest, indent=2))
