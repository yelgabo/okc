"""Build the M1 measurement file (not a candidate) from a gated M1 file
(tools/dl_gate_patch.py --m1 output), for a same-process bench against the release.

    python3 tools/dl_m1_bench_build.py GATED_M1.py OUT.py

- M1's fresh-input gate reports and never blocks: check 60cf58b69b75491e showed the M1-off
  control itself has order-1 element errors on fresh inputs (MoE output tails, routing flips at
  16 tokens), so as written the gate blocks every arm. M1 then runs as its timing tuner picks.
  M1_STRESS_REPLAYS drops to 1000 to keep the bench's prepare short.
- The fused MoE's L2 prefetch in its PDL preamble can be switched off per launch (bit 8 of the
  variant: Params::pdl 2 = PDL preamble without the prefetch hint), and _m1_tune times the whole
  step at decode 32 sliding and full on judge-like streams with the prefetch on, off and on
  again (the null), interleaved, cold L2 (coordinator's request, 2026-10-30: the layer MoE at
  32 tokens shows 10-14 us exposed against the public kernel while standalone it is 1.5-1.9 us
  behind; hypothesis: the prefetch of unit 0 of all 32 local experts, 18.9 MB, evicts data).
"""
import sys

PROBE = '''
MOE1_NOPF = False


def _judge_case(state, batch, length, layer, seed):
    """_timeline_case with reference.py's residual-stream distribution (so routing concentrates
    as on the judge) and local_start on the 32-expert range with the most routes."""
    data = _timeline_case(state, batch, length, layer, seed)
    tokens = batch * length
    dev = data["stream"].device
    g = torch.Generator().manual_seed(_REF_STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=g).to(dev)
    shared = torch.randn((_REF_STREAM_RANK, HIDDEN), generator=g).to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 7)
    coef = torch.randn((tokens, _REF_STREAM_RANK), generator=gen, device=dev)
    stream = (_REF_STREAM_MEAN * mean + (_REF_STREAM_SHARED / _REF_STREAM_RANK) ** 0.5
              * coef @ shared + (1 - _REF_STREAM_SHARED) ** 0.5
              * torch.randn((tokens, HIDDEN), generator=gen, device=dev))
    data["stream"].copy_(stream)
    data["attn_input"].copy_((stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True)
                                                   + RMS_EPS)).to(torch.bfloat16))
    saved = data["stream"].clone(), data["attn_input"].clone()
    ids = kernel_fn(data, state)[2]
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(),
                           minlength=N_EXPERTS // LOCAL_EXPERTS)
    data["local_start"].fill_(LOCAL_EXPERTS * int(pairs.argmax()))
    data["stream"].copy_(saved[0])
    data["attn_input"].copy_(saved[1])
    loc = ids.long().flatten() - LOCAL_EXPERTS * int(pairs.argmax())
    active = int(torch.unique(loc[(loc >= 0) & (loc < LOCAL_EXPERTS)]).numel())
    return data, active


def _moe_pf_probe(state, flush, rounds=11):
    """Whole step at decode 32 (32 x 1) sliding and full, two judge-like inputs each: the fused
    MoE's preamble prefetch on, off, and on again as the null, interleaved, cold L2; medians."""
    global MOE1_NOPF
    parts = []
    for layer in ("sliding", "full"):
        for seed in (9801, 9802):
            data, active = _judge_case(state, 32, 1, layer, seed)
            graphs = {}
            try:
                for arm, nopf in (("pf", False), ("nopf", True), ("pf2", False)):
                    MOE1_NOPF = nopf
                    graphs[arm] = _graph_build(lambda d=data: kernel_fn(d, state))
            finally:
                MOE1_NOPF = False
            med = _m1_pair(state, graphs, flush, rounds)
            del graphs
            parts.append(f"{layer[0]}{seed % 10} act {active} pf {med['pf']:.1f} nopf-pf "
                         f"{med['nopf'] - med['pf']:+.2f} null {med['pf2'] - med['pf']:+.2f}")
    return "moe prefetch d32 (us): " + "; ".join(parts)


'''


def main():
    src, out = sys.argv[1], sys.argv[2]
    text = open(src).read()
    reps = [
        ('            if not ok:\n                raise RuntimeError("M1 fresh-input gate: " + '
         'lines[-1])\n', "", 1),
        ("M1_STRESS_REPLAYS = 2000", "M1_STRESS_REPLAYS = 1000", 1),
        ("        if (tid == 0) {\n            for (int i = cta; i < p.nloc * 16; i += G) {",
         "        if (tid == 0 && p.pdl == 1) {\n            for (int i = cta; i < p.nloc * 16; "
         "i += G) {", 1),
        ("        if (tid == 0) {\n            for (int i = cta; i < p.nloc * 4; i += G) {",
         "        if (tid == 0 && p.pdl == 1) {\n            for (int i = cta; i < p.nloc * 4; "
         "i += G) {", 4),
        ("        p.pdl = pdl && m0 == 0 ? 1 : 0;",
         "        p.pdl = pdl && m0 == 0 ? 1 + static_cast<int>((variant >> 8) & 1) : 0;", 1),
        ("        moe1_dispatch(static_cast<int>(variant), p, sms, stream);",
         "        moe1_dispatch(static_cast<int>(variant & 255), p, sms, stream);", 1),
        ("POST_NORM_EPS, variant, pdl)", "POST_NORM_EPS, variant | (256 if MOE1_NOPF else 0), pdl)",
         1),
        ("def _m1_tune(state):", PROBE.lstrip("\n") + "def _m1_tune(state):", 1),
        ("    policy, v1s_saving = {}, 0.0\n",
         "    rep.append(_moe_pf_probe(state, flush))\n    policy, v1s_saving = {}, 0.0\n", 1),
        ("# okc: problem=muse_decode_layer\n",
         "# okc: problem=muse_decode_layer\n# M1 measurement build (not a candidate): gate "
         "report-only, MoE prefetch probe.\n", 1),
    ]
    for old, new, n in reps:
        if text.count(old) != n:
            raise SystemExit(f"expected {n} of {old[:70]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    open(out, "w").write(text)
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
