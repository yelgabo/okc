"""Offline conservative attention transform for attributed keell0renz layer fork.

Audited upstream 8e30e1be994db4310794b8d04e90d18d895deb84, layer SHA
1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a.
No device source/fence/math changes. One prepared state must not be replayed
concurrently; independent states have independent attention scratch.
"""
import ast


def _once(source, old, new):
    assert source.count(old) == 1, ('attention source drift', old[:100])
    return source.replace(old, new, 1)



E4M3_SOURCE = r'''def _f32_to_e4m3(x):
    """FP32 -> E4M3FN bits, RNE; match torch.to(float8_e4m3fn)."""
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = tl.abs(x)
    scaled = a * 512.0
    lower = scaled.to(tl.int32)
    fraction = scaled - lower.to(tl.float32)
    sub = lower + ((fraction > 0.5) | ((fraction == 0.5) & ((lower & 1) != 0))).to(tl.int32)
    ab = a.to(tl.uint32, bitcast=True)
    r = ab + 0x7FFFF + ((ab >> 20) & 1)
    e = (r >> 23).to(tl.int32) - 120
    m = (r >> 20) & 7
    normal = (e << 3) | m.to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    code = tl.where((a > 464.0) | (a != a), 0x7F, code)
    return code | sign.to(tl.int32)
'''

def patch_attention(source):
    source=source.replace('\r\n','\n')
    source=_once(source, '    st = {}\n    st["router_fp32"]', '    st = {}\n    st["attention_scratch"] = {"rows": {}, "cluster": {}, "retained": []}\n    st["router_fp32"]')
    source=_once(source, 'def _workspace7(n_rows, device):', 'def _workspace7(n_rows, device, owner):')
    source=_once(source, '    ws = _WS7.get(key)', '    ws = owner["rows"].get(key)')
    source=_once(source, '        n = max(n_rows, 1 << 15)', '        assert not torch.cuda.is_current_stream_capturing(), "attention rows must be warmed before capture"\n        n = max(n_rows, 1 << 15)')
    source=_once(source, '        _WS7[key] = ws', '        owner["rows"][key] = ws\n        owner["retained"].append(ws)  # old graph pointers remain live across growth/recapture')
    source=_once(source, 'def _cluster_bufs7(n_ctas, L, device):', 'def _cluster_bufs7(n_ctas, L, device, owner):')
    source=_once(source, '    cur = _CL7.get(key)', '    cur = owner["cluster"].get(key)')
    source=_once(source, '        cur = tuple(_guarded(t) for t in (', '        assert not torch.cuda.is_current_stream_capturing(), "attention cluster buffers must be warmed before capture"\n        cur = tuple(_guarded(t) for t in (')
    source=_once(source, '        _CL7[key] = cur', '        owner["cluster"][key] = cur\n        owner["retained"].append(cur)')
    source=_once(source, 'def _launch7(data, dbg=None, _mod=None, pdl=None, **override):', 'def _launch7(data, dbg=None, _mod=None, pdl=None, attention_state=None, **override):\n    assert attention_state is not None, "explicit prepared attention owner required"\n    owner = attention_state["attention_scratch"]')
    source=_once(source, '_workspace7(B * pairs * nc * L * 16, device)', '_workspace7(B * pairs * nc * L * 16, device, owner)')
    source=_once(source, '_cluster_bufs7(n_ctas if cs > 1 else 1, L, device)', '_cluster_bufs7(n_ctas if cs > 1 else 1, L, device, owner)')
    source=_once(source, '"k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)', '"k_scale": data["k_scale"], "v_scale": data["v_scale"]}, attention_state=state, **pdl)')
    # Both selected entry points must exclude unqualified scaled verify math.
    source=_once(source, 'agent_attn = state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)', 'agent_attn = state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg) and not (L == 8 and cfg.get("kv_scale", 1.0) != 1.0)')
    source=_once(source, 'if (plan is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)\n', 'if (plan is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)\n            and not (cfg["q_len"] == 8 and cfg.get("kv_scale", 1.0) != 1.0)\n')
    # Dispatch explicitly to captured base, not a later global kernel_fn wrapper.
    source=_once(source, '    st2 = dict(state)\n    st2["att_ok"] = False\n    return kernel_fn(data, st2)', '    if state.get("fa3") is None:\n        raise RuntimeError("conservative scaled-verify fallback requires FA3")\n    st2 = dict(state)\n    st2["att_ok"] = False\n    return _attention_fa3_base(data, st2)')
    source=_once(source, '\ndef _pick_expert_backend(', '\n_attention_fa3_base = kernel_fn  # stable base; avoids fallback recursion through later wrappers\n\ndef _pick_expert_backend(')
    # Base false branch must not silently run a different backend without FA3.
    source=_once(source, '    dev = data["attn_input"].device\n    x = data["attn_input"]\n    sliding = cfg["window"] > 0', '    if L == 8 and cfg.get("kv_scale", 1.0) != 1.0 and state.get("fa3") is None:\n        raise RuntimeError("conservative scaled-verify fallback requires FA3")\n    dev = data["attn_input"].device\n    x = data["attn_input"]\n    sliding = cfg["window"] > 0')
    # Preserve baseline BF16 QKV projection boundary before normalization/cache/gate.
    anchor = '    rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, 1) / 64.0 + 1e-5)[:, None]'
    casts = ''.join('    '+name+' = '+name+'.to(tl.bfloat16).to(tl.float32)\n' for name in ('qe','qo','ke','ko','v','g'))
    source=_once(source, anchor, casts+anchor)
    # Exact E4M3FN RNE, including signed subnormal ties and NaN overflow.
    # PyTorch baseline converts >464 and infinities to signed NaN, not saturation.
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_f32_to_e4m3')
    lines=source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=[E4M3_SOURCE+'\n']
    source=''.join(lines)
    source=_once(source, '    return _after_attention(data, state, o_in, gate, T)\n', '    # FA3 output rounds to BF16 before official per-head norm/gate.\n    a = o_in.to(torch.bfloat16).view(T, 16, 64).float()\n    a = a * torch.rsqrt(a.square().mean(dim=-1, keepdim=True) + 1e-5)\n    o_in = (a * torch.sigmoid(gate.view(T, 16, 64).float())).to(torch.bfloat16).view(T, Q_DIM)\n    return _after_attention(data, state, o_in, None, T)\n')
    ast.parse(source)
    return source
