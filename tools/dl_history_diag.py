"""Build the input-history diagnostic from a layer file (diagnostic, not a candidate).

    python3 tools/dl_history_diag.py LAYER.py OUT.py

Check 60cf58b69b75491e: at decode 1 sliding the attention output of an eager call on a fresh
input differs from that input's reference by up to 1e-4 (every arm, eager included), while the
attention stress on repeated inputs is bitwise (replayed_vs_eager 0.0 in fdf75ba5a8124544).
Merge-order noise would show on repeated inputs too; a dependence on the previous call's input
(state carried between calls: the cache rows the step writes, or the attention workspace)
would not. After prepare, per scored T <= 32 shape and for qkv, attention, the whole step and
the MoE alone, this measures with eager calls:

  rep   the largest relative difference between two calls on input k, each right after a
        call on input k (repeated-input noise)
  hist  the largest relative difference between a call on input k right after a call on input
        j != k and the repeated-input result (history dependence)
  sep   as hist, but the call on input j runs on a second copy of the case (its own cache and
        inputs, the same layer state): hist with sep near 0 puts the carrier in the case's own
        buffers (cache), hist with sep near hist in the shared state (workspace)

hist well above rep means an output depends on the previous call's input.
"""
import sys

DIAG = '''

def _hist_probe(run, load, n=4, load_other=None):
    rep = hist = sep = 0.0
    for k in range(n):
        load(k)
        run()
        load(k)
        o2 = run()
        load(k)
        o3 = run()
        rep = max(rep, _rel(o2, o3))
        for j in range(n):
            if j == k:
                continue
            load(j)
            run()
            load(k)
            hist = max(hist, _rel(run(), o3))
            if load_other is not None:
                load_other(j)
                load(k)
                sep = max(sep, _rel(run(), o3))
    return rep, hist, sep


def _history_diag(state):
    lines = []
    for batch, length, layer in ((1, 1, "sliding"), (1, 1, "full"), (1, 8, "sliding"),
                                 (1, 8, "full"), (16, 1, "sliding"), (16, 1, "full"),
                                 (32, 1, "sliding"), (32, 1, "full")):
        tag = f"{batch}x{length}{layer[0]}"
        try:
            data = _timeline_case(state, batch, length, layer, 9300 + batch * length)
            other = _timeline_case(state, batch, length, layer, 9300 + batch * length)
            dev = data["stream"].device
            gen = torch.Generator(device=dev).manual_seed(777 + batch * length)
            bank_s = torch.randn((4, batch * length, HIDDEN), generator=gen, device=dev)
            bank_a = (bank_s * torch.rsqrt(bank_s.pow(2).mean(-1, keepdim=True) + RMS_EPS)
                      ).to(torch.bfloat16)
            cfg = data["config"]
            path = _attn_path(cfg, state)
            nq = state["dense"]["qkv_gate"].shape[0]
            qkvs = [_proj(bank_a[k], "qkv_gate", nq, state).clone() for k in range(4)]
            cur = {}

            def load_qkv(k):
                cur["q"] = qkvs[k]

            def load_step(k):
                data["stream"].copy_(bank_s[k])
                data["attn_input"].copy_(bank_a[k])

            def other_attn(j):
                path(qkvs[j], other, state)

            def other_step(j):
                other["stream"].copy_(bank_s[j])
                other["attn_input"].copy_(bank_a[j])
                kernel_fn(other, state)

            q = _hist_probe(lambda: _proj(data["attn_input"], "qkv_gate", nq, state).clone(),
                            load_step)
            a = _hist_probe(lambda: path(cur["q"], data, state).clone(), load_qkv,
                            load_other=other_attn)
            s = _hist_probe(lambda: kernel_fn(data, state)[0].clone(), load_step,
                            load_other=other_step)
            torch.cuda.synchronize()
            fmt = lambda r: "/".join(f"{v:.0e}" for v in r)  # noqa: E731
            lines.append(f"HIST {tag} rep/hist/sep qkv {fmt(q[:2])} attn {fmt(a)} step {fmt(s)}")
        except Exception as exc:
            lines.append(f"HIST {tag} error {type(exc).__name__}: {str(exc)[:160]}")
    for tokens in (1, 8, 16, 32, 128, 256):
        try:
            bank = [_moe_judge_routing(tokens, _TUNE_Q_LEN[tokens], 5100 + 13 * k + tokens, state)
                    for k in range(4)]
            opts = _moe_opts(tokens, state)
            cur = {}

            def load_moe(k):
                cur["b"] = bank[k]

            m = _hist_probe(lambda: _experts(*cur["b"], state, opts).clone(), load_moe)
            v = opts.get("fused")
            lines.append(f"HIST moe T={tokens} {'w4' if v is None else _moe1_name(v)} rep/hist "
                         f"{m[0]:.0e}/{m[1]:.0e}")
        except Exception as exc:
            lines.append(f"HIST moe T={tokens} error {type(exc).__name__}: {str(exc)[:160]}")
    return lines


_hist_inner_prepare = prepare


def prepare(weights):
    state = _hist_inner_prepare(weights)
    lines = _history_diag(state)
    for line in lines:
        print(line, flush=True)
    return state
'''


def main():
    src, out = sys.argv[1], sys.argv[2]
    text = open(src).read()
    reps = [
        ("TIMELINE_SHAPES = (", "TIMELINE_SHAPES = () and (", 1),
        ("# okc: problem=muse_decode_layer\n",
         "# okc: problem=muse_decode_layer\n# Input-history diagnostic (not a candidate).\n", 1),
    ]
    for old, new, n in reps:
        if text.count(old) != n:
            raise SystemExit(f"expected {n} of {old[:60]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    open(out, "w").write(text + DIAG)
    print("wrote", out, len(text) + len(DIAG))


if __name__ == "__main__":
    main()
