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
    ast.parse(source)
    return source
