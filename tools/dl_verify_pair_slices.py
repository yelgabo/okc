"""Explicit, private G/owned slices for verify16/32, sliding/full context.

Build from the already reviewed single-cell slices. Only their explicit Python
preparation/scope changes; selected CUDA literal values remain byte-identical.
No generation path dispatches a GPU job or changes a production file.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import dl_g_verify_slice as graft
import dl_owned_slice as owned
from dl_owned_main_slice import specialize


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(("verify slice anchor", old, text.count(old)))
    return text.replace(old, new)


def graft_custom():
    text = once(graft.CUSTOM, "_GRAFT_LT_T = (256,)", "_GRAFT_LT_T = (128, 256)")
    text = once(text,
        'if set(state["graft_lt"]) != {(name, 256) for name in ("qkv", "o", "pr", "un")}:',
        'if set(state["graft_lt"]) != {(name, m) for name in ("qkv", "o", "pr", "un") for m in (128, 256)}:')
    start = text.index('    expected = {"batch": 32,')
    end = text.index('    device = data["attn_input"].device', start)
    text = text[:start] + '''    batch = config.get("batch")
    window = config.get("window")
    if batch not in (16, 32) or window not in (0, 2048):
        raise RuntimeError("private G arm supports verify16/32 sliding/full only")
    t = batch * 8
    nw = 32 if t == 128 else 16
    expected = {"batch": batch, "q_len": 8, "tokens": t, "window": window,
                "page_size": 1, "hidden": 8192, "num_q_heads": 16,
                "num_kv_heads": 2, "head_dim": 64, "top_k": 8,
                "n_experts": 256, "local_experts": 32, "kv_scale": 1.0}
    if any(config.get(name) != value for name, value in expected.items()):
        raise RuntimeError("private G verify arm geometry drift")
    if data["page_table"].shape != (batch, 86008):
        raise RuntimeError("private G verify arm page capacity drift")
''' + text[end:]
    for post in ("False", "True"):
        text = once(text,
            f'8192, 8192, {post}, 4, 16,\n        num_warps=16, launch_pdl=False)',
            f'8192, 8192, {post}, 4, nw,\n        num_warps=nw, launch_pdl=False)')
    return text


OWNED_LT = '''def _slice_lt(state):
    lt = _dlplt_build()
    state["dlplt"] = {}
    for layer, m, layout, cfg in (
        ("unpooler", 128, 3, [66,18,1,0,0,2,35,0,5]),
        ("unpooler", 256, 1, [66,20,1,0,0,4,35,0,3]),
        ("qkv_gate", 256, 0, [66,124,1,0,0,0,35,0,2])):
        w = state["dense"][layer]; n,k = w.shape
        ws = torch.zeros(32 << 20,dtype=torch.uint8,device=w.device)
        wa = w.t().contiguous() if layout % 2 else w
        plan = lt.lt_plan_new(m,n,k,layout,ws.numel())
        idx = lt.lt_add_config(plan,list(cfg)) if plan >= 0 else -1
        assert idx >= 0, ("verify slice Lt pin rejected", layer, m)
        def run(x, plan=plan, idx=idx, wa=wa, ws=ws, n=n, m=m):
            out = torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
            rc = lt.lt_run(plan,idx,wa,x,out,ws)
            if rc: raise RuntimeError(f"verify slice Lt status {rc}")
            return out
        state["dlplt"][layer,m] = run
'''


def owned_custom():
    text = owned.CUSTOM
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_slice_lt")
    text = once(text, ast.get_source_segment(text, node), OWNED_LT.rstrip())
    text = once(text,
        '''    entry = state["dlplt"].get((layer, x.shape[0]))''',
        '''    if layer == "qkv_gate" and x.shape[0] == 128:
        return state["verify_qkv128"](x.contiguous())
    entry = state["dlplt"].get((layer, x.shape[0]))''')
    text = once(text,
        '''      "ffn_boundary":FusedPostNormResidualPreNorm(postffn,fr,nxt),''',
        '''      "attn_boundary":FusedPostNormResidualPreNorm(post,ar,pre),
      "ffn_boundary":FusedPostNormResidualPreNorm(postffn,fr,nxt),''')
    text = once(text, '''    _ext(); _one_setup(state)''',
        '''    _ext(); _one_setup(state)
    state["verify_diag"] = torch.zeros(4,dtype=torch.int32,device=device)
    state["verify_qkv128"] = _skp_wd_runner(_skp_load(),"b1n128c3s8h",
        state["dense"]["qkv_gate"],state["verify_diag"])''')
    text = once(text,
        '''    assert _one_plan({"batch":32,"q_len":8,"page_size":1,"window":2048},86008,
                     state["one"]["resident"]) == (4,1,3), "slice attention selection mismatch"''',
        '''    for batch in (16,32):
        for window in (0,2048):
            expected = (264,1,51) if not window else (8 if batch==16 else 4,1,3)
            assert _one_plan({"batch":batch,"q_len":8,"page_size":1,"window":window},86008,
                state["one"]["resident"]) == expected, "verify slice attention selection drift"''')
    text = once(text,
        'OWNED_SLICE source-only policy verify32/sliding ONE(4,1,3), MOE51, Lt L0/L1; no autotuning',
        'OWNED_VERIFY_SLICE verify16/32 sliding/full, QKV128 WD, SG128 boundary, MOE51, three Lt pins; no autotuning')
    start = text.index('    assert all(c[k]==v for k,v in {"batch":32,')
    end = text.index('    qkv=_slice_proj', start)
    text = text[:start] + '''    batch=c.get("batch"); window=c.get("window")
    assert batch in (16,32) and window in (0,2048), "unsupported verify slice config"
    assert all(c.get(k)==v for k,v in {"batch":batch,"q_len":8,"tokens":batch*8,"page_size":1,
        "window":window,"hidden":8192,"num_q_heads":16,"num_kv_heads":2,"head_dim":64,
        "n_experts":256,"top_k":8,"local_experts":32,"kv_scale":1.0}.items()), "verify slice geometry drift"
    assert data["page_table"].shape==(batch,86008), "unsupported verify slice context capacity"
''' + text[end:]
    text = once(text,
        '''_abnd(data["stream"],attn_output,False,state,"c1x512")''',
        '''_abnd(data["stream"],attn_output,False,state,"sglang" if batch==16 else "c1x512")''')
    return text


def literals(text):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(text).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            and len(n.value.value) > 1000}


def build(root):
    owned_source = (root / "muse_decode_layer/cand/layer_lt_boundary.py").read_text(encoding="utf8")
    g, _ = graft.build((root / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8"))
    o, _ = owned.build(owned_source, custom_source=owned_custom())
    g = once(g, graft.CUSTOM, graft_custom())
    o, _, _ = specialize(o)
    o = once(o, "CHECK-ONLY single-load diagnostic", "CHECK-ONLY four-cell verify diagnostic")
    for old, new in ((graft.build((root / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8"))[0], g),
                     (specialize(owned.build((root / "muse_decode_layer/cand/layer_lt_boundary.py").read_text(encoding="utf8"))[0])[0], o)):
        original_literals, revised_literals = literals(old), literals(new)
        assert all(revised_literals.get(name) == value for name, value in original_literals.items()), "selected native literal drift"
    full_literals = literals(owned_source)
    assert set(literals(o)) - set(literals(specialize(owned.build(owned_source)[0])[0])) == {
        "_SKP_SRC", "_SKP_GEMV_SRC", "_SKP_WD_SRC"}
    assert all(literals(o)[name] == full_literals[name] for name in ("_SKP_SRC", "_SKP_GEMV_SRC", "_SKP_WD_SRC"))
    arms = {"G": g, "owned": o}
    manifest = {"scope": "verify16/32 sliding/full, page1, kv_scale1, capacity86008",
                "arms": {name: {"bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()}
                         for name, text in arms.items()},
                "native_literals_unchanged": True, "H200_qualified": False}
    return arms, manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path); p.add_argument("output", type=Path)
    args = p.parse_args()
    arms, manifest = build(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in arms.items():
        args.output.joinpath(name + "-verify-arm.py").write_bytes(source.encode())
    args.output.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps(manifest, indent=2))
