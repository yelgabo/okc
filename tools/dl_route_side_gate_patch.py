"""Move the route side-stream self-test onto the fresh-input gate (user decision 2026-10-31,
relayed by the coordinator; docs/runs-decode-layer.md, "Validation policy change"). The old
test compared one MoE call with its routing metadata built on the side stream against the
one-stream call, eagerly and as a graph, under a 1e-4 bound that the MoE's FP32 atomics trip
on noise (b70cfa48c0264d20: 1.21e-4, and the fallback costs about 1.7 us at d1 sliding).

The gated test, per token count (1, 8, 16, 32, 128, 256): GATE_BANK fresh routings (inputs,
top-8 weights and ids), each run once on the one-stream path eagerly as its reference; the
candidate is a CUDA graph of the side-stream step that copies the next bank entry into the
captured inputs and compares its output with that entry's reference on the device, replayed
ROUTE_STRESS times; the control is the same side-stream step made eagerly as often.
_gate_verdict decides (order-1 = relative L2 >= 1e-2, the rest against the control's
envelope); the negative controls run report-only on an eager output, as for the route variant
gate. A failure raises, so route_side_ok stays False as before.

    python3 tools/dl_route_side_gate_patch.py IN.py OUT.py "HEADER LINE 2"
"""
import sys

NEW = '''def _route_side_self_test(state):
    """The MoE call with its routing metadata built on the side stream and joined right before
    GEMM1, as kernel_fn runs it, under the fresh-input gate (user decision 2026-10-31): per
    token count, GATE_BANK routings each run once on the one-stream path as references; the
    side-stream step captured in a graph that loads the next routing every replay (candidate)
    and the same step made eagerly (control), ROUTE_STRESS calls each, compared on the device.
    Raises on a gate failure."""
    side = state["side_stream"]
    lines = []
    for tokens in (1, 8, 16, 32, 128, 256):
        opts = _moe_opts(tokens, state) if state["moe_ours"] else None
        bank = [_moe_routing(tokens, 3000 + tokens + 1009 * k, 96, True)
                for k in range(GATE_BANK)]
        local_start = bank[0][3]
        xb = torch.stack([b[0] for b in bank])
        twb = torch.stack([b[1] for b in bank])
        idb = torch.stack([b[2] for b in bank])
        x, tw, ids = xb[0].clone(), twb[0].clone(), idb[0].clone()
        dev = x.device
        ys = []
        for k in range(GATE_BANK):
            x.copy_(xb[k])
            tw.copy_(twb[k])
            ids.copy_(idb[k])
            ys.append(_experts(state["pre_expert_norm"](x) if state["norm_ours"] else x, tw,
                               ids, local_start, state, opts).clone())
        refs = [_gate_refs(ys)]
        res = {}

        def run():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            # The main-stream kernel producing GEMM1's input: our norm (a PDL primary and
            # secondary) when enabled, else a stand-in for torch's.
            xin = _pre_norm(x, state) if state["norm_ours"] else x * 1
            with torch.cuda.stream(side):
                meta = _route_meta(tw, ids, local_start, tokens, state, opts)
            main.wait_stream(side)
            if not torch.cuda.is_current_stream_capturing():
                for tensor in _meta_tensors(meta):
                    tensor.record_stream(main)
            res["out"] = _experts_run(xin, meta, state)

        sel = torch.zeros(1, dtype=torch.long, device=dev)

        def step(acc):
            sel.add_(1)
            i = sel % GATE_BANK
            x.copy_(xb.index_select(0, i)[0])
            tw.copy_(twb.index_select(0, i)[0])
            ids.copy_(idb.index_select(0, i)[0])
            run()
            _gate_add(acc, [res["out"]], refs, i)

        cand, ctrl = _gate_acc(1, dev), _gate_acc(1, dev)
        graph = _graph_build(lambda: step(cand))
        _gate_zero(cand)
        for _ in range(ROUTE_STRESS):
            graph.replay()
        torch.cuda.synchronize()
        del graph
        for _ in range(ROUTE_STRESS):
            step(ctrl)
        x.copy_(xb[0])
        tw.copy_(twb[0])
        ids.copy_(idb[0])
        run()
        obs = res["out"].clone()
        torch.cuda.synchronize()
        neg, misses = _gate_negctl(obs, refs[0], 0, 1, ctrl, 191 + tokens)
        state.setdefault("gate_negctl", {})[f"route side@{tokens}"] = neg
        if misses:
            _gate_fail("sensitivity", f"route side@{tokens}: negative controls {misses}")
        ok, text, classes = _gate_verdict(cand, ctrl)
        state.setdefault("route_gate", {})[f"side@{tokens}"] = (
            f"{'ok' if ok else 'FAIL'} {_gate_short(cand, ctrl)}")
        lines.append(f"T={tokens} {'ok' if ok else 'FAIL'} {_gate_short(cand, ctrl)}")
        print(f"route-side gate T={tokens}: {text}")
        if not ok:
            raise RuntimeError(f"[{','.join(classes)}] route-side fresh-input gate T={tokens}: "
                               f"{text}")
    state["route_side_report"] = lines


'''

SUMMARY_OLD = '''    lines.append(_gate_manifest(state))'''
SUMMARY_NEW = '''    for item in state.get("route_side_report", []):
        lines.append(f"route side gate {item}")
    lines.append(_gate_manifest(state))'''


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, out, header = sys.argv[1:]
    text = open(src).read()
    start = "def _route_side_self_test(state):"
    end = "def _norm_self_test(state):"
    if text.count(start) != 1 or text.count(end) != 1 or "route_side_report" in text:
        raise SystemExit("IN has no single unpatched _route_side_self_test before _norm_self_test")
    a, b = text.index(start), text.index(end)
    old = text[a:b]
    if "if not worst < 1e-4:" not in old:
        raise SystemExit("the route side-stream test is not the 1e-4 version")
    text = text[:a] + NEW + text[b:]
    if text.count(SUMMARY_OLD) != 1:
        raise SystemExit("the SUMMARY manifest line is not where expected")
    text = text.replace(SUMMARY_OLD, SUMMARY_NEW)
    lines = text.split("\n")
    if lines[0] != "# okc: problem=muse_decode_layer":
        raise SystemExit("line 1 is not the okc header")
    lines[1] = header
    open(out, "w").write("\n".join(lines))
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
