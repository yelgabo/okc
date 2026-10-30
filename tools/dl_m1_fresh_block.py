def _m1_fresh_shape(state, batch, length, layer, policy, bank_n=5):
    """Every replay draws new rows: a device counter picks one of bank_n (stream, attn_input)
    pairs, copied into the captured inputs before kernel_fn, and the replay's outputs are
    compared on the device with the M1-off eager outputs of that pair; the largest relative
    difference over all replays and the count of replays whose ids differ are kept. A read
    of the previous replay's rows gives a difference of order 1, not the atomics' order."""
    data = _timeline_case(state, batch, length, layer, 9300 + batch * length)
    dev = data["stream"].device
    gen = torch.Generator(device=dev).manual_seed(777 + batch * length)
    tokens = batch * length
    bank_s = torch.randn((bank_n, tokens, HIDDEN), generator=gen, device=dev)
    bank_a = (bank_s * torch.rsqrt(bank_s.pow(2).mean(-1, keepdim=True) + RMS_EPS)
              ).to(torch.bfloat16)
    refs_h, refs_s, refs_i = [], [], []
    state["m1_policy"] = {}
    for k in range(bank_n):
        data["stream"].copy_(bank_s[k])
        data["attn_input"].copy_(bank_a[k])
        out = kernel_fn(data, state)
        refs_h.append(out[0].clone())
        refs_s.append(out[1].clone())
        refs_i.append(out[2].clone())
    ref_h, ref_s, ref_i = torch.stack(refs_h), torch.stack(refs_s), torch.stack(refs_i)
    nh = ref_h.float().flatten(1).norm(dim=1)
    ns = ref_s.float().flatten(1).norm(dim=1)
    ctr = torch.zeros(1, dtype=torch.long, device=dev)
    errmax = torch.zeros(2, dtype=torch.float32, device=dev)
    bad_ids = torch.zeros(1, dtype=torch.long, device=dev)
    over = torch.zeros(2, dtype=torch.long, device=dev)
    state["m1_policy"] = policy

    def run():
        ctr.add_(1)
        i = ctr % bank_n
        data["stream"].copy_(bank_s.index_select(0, i)[0])
        data["attn_input"].copy_(bank_a.index_select(0, i)[0])
        out = kernel_fn(data, state)
        eh = (out[0].float() - ref_h.index_select(0, i)[0].float()).norm() / nh.index_select(0, i)[0]
        es = (out[1].float() - ref_s.index_select(0, i)[0].float()).norm() / ns.index_select(0, i)[0]
        e2 = torch.stack((eh, es))
        torch.maximum(errmax, e2, out=errmax)
        em = e2.max()
        over.add_(torch.stack(((em > 1e-4).long(), (em > 1e-2).long())))
        bad_ids.add_((out[2] != ref_i.index_select(0, i)[0]).any().long())

    graph = _graph_build(run)
    errmax.zero_()
    bad_ids.zero_()
    over.zero_()
    for _ in range(M1_STRESS_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    del graph
    state["m1_policy"] = {}
    e = errmax.tolist()
    o = over.tolist()
    return f"{e[0]:.1e}/{e[1]:.1e} >1e-4:{o[0]} >1e-2:{o[1]} ids!={int(bad_ids)}"


def _m1_fresh(state):
    on = {b: "o+front" for b in M1_BUCKETS}
    lines = []
    for batch, length, layer in ((1, 8, "full"), (1, 8, "sliding"), (1, 1, "full"),
                                 (1, 1, "sliding"), (16, 1, "full"), (32, 1, "full")):
        try:
            reps = 3 if (batch, length, layer) == (1, 8, "full") else 1
            offs, ons = [], []
            for _ in range(reps):
                offs.append(_m1_fresh_shape(state, batch, length, layer, {}))
                ons.append(_m1_fresh_shape(state, batch, length, layer, on))
            lines.append(f"fresh {batch}x{length}{layer[0]} off {', '.join(offs)} | on "
                         f"{', '.join(ons)}")
        except Exception as exc:
            lines.append(f"fresh {batch}x{length}{layer[0]} error {type(exc).__name__}: "
                         f"{str(exc)[:150]}")
    return lines


