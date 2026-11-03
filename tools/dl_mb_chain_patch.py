"""M-B slices 1-3 on layer_m1rt51s (archive/docs/plan-2026-10-31-m-b-pdl-chain.md). M1's mode string
becomes a set of '+'-joined switches, each tunable per bucket by the stable M1 tuner:

  o      the o projection as a PDL secondary of attention (M1, unchanged)
  front  the skf front as a PDL secondary of the attention boundary (M1, unchanged)
  unp    the unpooler sk projection as a PDL secondary of the MoE (slice 1)
  trig   every sk / skf launch triggers its dependents at entry (slice 2); the fused MoE
         kernels trigger after their grid barrier in every mode (inert unless the next launch
         carries the PDL attribute, which only unp gives it)
  attn   attention as a PDL secondary of qkv at 9-32 tokens, on non-cooperative modes only
         (slice 3)

Idea learned from the F-LAYER-0 comparison (four-arm checks 20abe6291eb04ed6,
00607fc6f8c84c77) of alygg77/okc-muse-spark-kernels @ 6cae49a, which chains its glue and
projection kernels by PDL at T <= 32. Re-implemented on this layer's own M1 machinery; no
donor code.

Ordering (plan, "Trigger-order invariant"): the MoE triggers after its own
griddepcontrol.wait (the grid barrier follows it), so "the MoE has triggered" still implies
"route, front and the attention boundary are complete", which the FFN boundary's pre-wait read
of the stream relies on. An sk kernel triggers at entry, before its wait; that is safe because
no successor of an sk launch reads, before its own wait, anything written after the sk
kernel's own primary started: the attention boundary reads the step's input stream and state,
route reads prepare-time weights, the FFN boundary's two-back read is covered by the MoE's
post-wait trigger, and attention (after qkv) reads harness K/V rows and page tables.

    python3 tools/dl_mb_chain_patch.py IN.py OUT.py "HEADER LINE 2" [--round=3]
"""
import sys

SK_TRIG_NOTE = ("    // M-B trig (bit 29 of Mf, tools/dl_mb_chain_patch.py): no CTA of this kernel waits on\n"
                "    // another, so the dependents may launch at once; they still wait for completion.\n")

DEVICE = [
    ('''    const int M = Mf & 0x3fffffff;
''',
     '''    const int M = Mf & 0x1fffffff;
#if SK_PDL
''' + SK_TRIG_NOTE + '''    if ((Mf >> 29) & 1) {
        asm volatile("griddepcontrol.launch_dependents;");
    }
#endif
''', 1),
    ('''bool sk_pdl_launch = false;
''',
     '''bool sk_pdl_launch = false;
bool sk_trig_launch = false;
#define SK_MF(M) ((M) | (sk_trig_launch ? (1 << 29) : 0))
#else
#define SK_MF(M) (M)
''', 1),
    ('''                                              M | (1 << 30), N, K, diag, rws, rpart,''',
     '''                                              SK_MF(M) | (1 << 30), N, K, diag, rws, rpart,''',
     1),
    ('''        kern<<<N / C::R, C::THREADS, smem, stream>>>(xmap, wsp, op, tp, pmax, base4, M, N, K, diag, rws, rpart,''',
     '''        kern<<<N / C::R, C::THREADS, smem, stream>>>(xmap, wsp, op, tp, pmax, base4, SK_MF(M), N, K, diag, rws, rpart,''',
     1),
    ('''    sk_pdl_launch = ((cfg >> 16) & 1) != 0;
''',
     '''    sk_pdl_launch = ((cfg >> 16) & 1) != 0;
    sk_trig_launch = ((cfg >> 17) & 1) != 0;
''', 2),
    ('''    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[6] = gtimer();
    }
''',
     '''    __syncthreads();
    // M-B: past the grid barrier no CTA waits on another, and this kernel's own wait came
    // before it, so a dependent that launches now cannot see an incomplete predecessor chain.
    asm volatile("griddepcontrol.launch_dependents;");
    if (kStamp && tid == 0) {
        S.dg[6] = gtimer();
    }
''', 4),
    ('''    __syncthreads();
    if constexpr (kDiag) {
        dt[3] = gtimer();
    }
''',
     '''    __syncthreads();
    // M-B: past the grid barrier no CTA waits on another, and this kernel's own wait came
    // before it, so a dependent that launches now cannot see an incomplete predecessor chain.
    asm volatile("griddepcontrol.launch_dependents;");
    if constexpr (kDiag) {
        dt[3] = gtimer();
    }
''', 1),
    ('''    if (q_len == L && mode == M) { one_launch<L, M>(p, gx, gy, stream, coop, pdl); hit = true; }''',
     '''    if (q_len == L && mode == M) { one_launch<L, M>(p, gx, gy, stream, coop, pdl && !coop); hit = true; }''',
     1),
]

PY = [
    ('''SK_PDL_BIT = 1 << 16
''',
     '''SK_PDL_BIT = 1 << 16
# M-B trig: the sk / skf launch triggers its dependents at entry (tools/dl_mb_chain_patch.py).
SK_TRIG_BIT = 1 << 17
''', 1),
    ('''    def run(x, pdl=False):
        # Bit 16 of the config: a PDL launch (built with SK_PDL, M1); without the switch the
        # extension masks nothing and would reject the config, so it is set only then.
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        if pdl:
            _M1_PDL_LAUNCHES[0] += 1
        ext.sk_fwd(cid | (SK_PDL_BIT if pdl else 0), stream, x, out, tab, pmax, base4, n, k, diag)''',
     '''    def run(x, pdl=False, trig=False):
        # Bit 16 of the config: a PDL launch (built with SK_PDL, M1); without the switch the
        # extension masks nothing and would reject the config, so it is set only then.
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        if pdl:
            _M1_PDL_LAUNCHES[0] += 1
        ext.sk_fwd(cid | (SK_PDL_BIT if pdl else 0) | (SK_TRIG_BIT if trig else 0), stream, x,
                   out, tab, pmax, base4, n, k, diag)''', 1),
    ('''        def run(x, pdl=False, cid=cid, stream=stream, rstream=rstream, tab=tab, pmax=pmax,
                base4=base4, rpart=rpart, rcount=rcount):''',
     '''        def run(x, pdl=False, trig=False, cid=cid, stream=stream, rstream=rstream, tab=tab,
                pmax=pmax, base4=base4, rpart=rpart, rcount=rcount):''', 1),
    ('''            ext.skf_fwd(cid | (SK_PDL_BIT if pdl else 0), stream, rstream, x, out, rout, tab,''',
     '''            ext.skf_fwd(cid | (SK_PDL_BIT if pdl else 0) | (SK_TRIG_BIT if trig else 0),
                        stream, rstream, x, out, rout, tab,''', 1),
    ('''            if layer == "o" and getattr(pick[1], "sk", False) and _m1_mode(m, state) != "off":
                return pick[1](x.contiguous(), True)''',
     '''            if getattr(pick[1], "sk", False) and _m1_mode(m, state) != "off":
                return pick[1](x.contiguous(), _m1_pdl(layer, m, state),
                               _m1_has(m, state, "trig"))''',
     1),
    ('''_m1_mode(tokens, state) == "o+front")''',
     '''_m1_has(tokens, state, "front"), _m1_has(tokens, state, "trig"))''', 2),
    ('''    """M1's choice for this token count: "off", "o" or "o+front"."""''',
     '''    """M1's choice for this token count: "off" or switches joined by "+" (o, front, unp,
    trig, attn; tools/dl_mb_chain_patch.py)."""''', 1),
    ('''M1_STRESS_REPLAYS = 2000
''',
     '''def _m1_has(tokens, state, switch):
    return switch in _m1_mode(tokens, state).split("+")


def _m1_pdl(layer, tokens, state):
    """Whether this sk projection launches as a PDL secondary: o under "o", the unpooler
    (whose predecessor is the MoE) under "unp"."""
    return _m1_has(tokens, state, {"o": "o", "unpooler": "unp"}.get(layer, "-"))


# M1's tuner modes; the gate replays the last, which makes every PDL launch any mode makes.
M1_MODES = ("off", "o", "o+front", "o+front+trig", "o+front+unp+trig", "o+front+unp+trig+attn")
M1_GATE_MODE = M1_MODES[-1]
M1_STRESS_REPLAYS = 2000
'''),
    ('''    on = {b: "o+front" for b in M1_BUCKETS}''',
     '''    on = {b: M1_GATE_MODE for b in M1_BUCKETS}''', 1),
    ('''            a, b = pick[1](x, False), pick[1](x, True)
            torch.cuda.synchronize()
            if not torch.equal(a, b):
                raise RuntimeError(f"o {pick[0]} m={m}: PDL launch differs from the plain one")
            checked.append(f"o:{pick[0]}@{m}")''',
     '''            a, b = pick[1](x, False), pick[1](x, True, True)
            torch.cuda.synchronize()
            if not torch.equal(a, b):
                raise RuntimeError(f"o {pick[0]} m={m}: PDL launch differs from the plain one")
            checked.append(f"o:{pick[0]}@{m}")
        pick = state["skp_pick"].get(("unpooler", m))
        if pick is not None and getattr(pick[1], "sk", False):
            x = torch.randn((m, state["dense"]["unpooler"].shape[1]), generator=gen, device=dev)
            x = x.to(torch.bfloat16)
            a, b = pick[1](x, False), pick[1](x, True, True)
            torch.cuda.synchronize()
            if not torch.equal(a, b):
                raise RuntimeError(f"unpooler {pick[0]} m={m}: PDL launch differs")
            checked.append(f"unp:{pick[0]}@{m}")''', 1),
    ('''            (a, ar), (b, br) = fused(x, False), fused(x, True)''',
     '''            (a, ar), (b, br) = fused(x, False), fused(x, True, True)''', 1),
    ('''        st["coop_launch"], st["pdl"], st["dbg"] if dbg else st["nodbg"])''',
     '''        st["coop_launch"], st["pdl"] or _attn_pdl(batch * length, state),
        st["dbg"] if dbg else st["nodbg"])


def _attn_pdl(tokens, state):
    """M-B attn: attention as a PDL secondary of qkv at 9-32 tokens. The host drops the
    attribute on cooperative modes; d16 / d32 plan non-cooperative ones (1 and 49). One token
    stays plain: its qkv is cgemv, which sets no trigger, and the census would not see it."""
    if 8 < tokens <= M1_BUCKETS[-1] and _m1_has(tokens, state, "attn"):
        _M1_PDL_LAUNCHES[0] += 1
        return True
    return False''', 1),
]

MODES_OLD = '("off", "o", "o+front")'

# --round=3 (after a3760918c13c4e40): the sk entry trigger on every launch cost 1.0 us at 8
# tokens and 5.5-6.1 us at 16 / 32 in the stable M1 tuner's paired whole-step rounds, while
# unp + trig saved 1.39 us at 8 (9 of 9 rounds). Round 3 drops the blanket trigger from the
# tuner's modes, adds utrig (the trigger on the unpooler launch only, whose successor is the
# FFN boundary) and times unp and attn without it. The gate replays every switch at once.
ROUND3 = [
    ('''M1_MODES = ("off", "o", "o+front", "o+front+trig", "o+front+unp+trig", "o+front+unp+trig+attn")
M1_GATE_MODE = M1_MODES[-1]''',
     '''M1_MODES = ("off", "o", "o+front", "o+unp", "o+front+unp", "o+front+unp+utrig", "o+unp+attn")
M1_GATE_MODE = "o+front+unp+utrig+trig+attn"'''),
    ('''                return pick[1](x.contiguous(), _m1_pdl(layer, m, state),
                               _m1_has(m, state, "trig"))''',
     '''                return pick[1](x.contiguous(), _m1_pdl(layer, m, state),
                               _m1_has(m, state, "trig") or (
                                   layer == "unpooler" and _m1_has(m, state, "utrig")))'''),
    ('''    """M1's choice for this token count: "off" or switches joined by "+" (o, front, unp,
    trig, attn; tools/dl_mb_chain_patch.py)."""''',
     '''    """M1's choice for this token count: "off" or switches joined by "+" (o, front, unp,
    trig, utrig, attn; tools/dl_mb_chain_patch.py)."""'''),
]


def apply(text, reps, what):
    for old, new, n in reps:
        if text.count(old) != n:
            raise SystemExit(f"{what}: expected {n} of {old[:70]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = [a for a in sys.argv[1:] if a.startswith("--")]
    if len(args) != 3 or not set(opts) <= {"--round=3"}:
        raise SystemExit(__doc__)
    src, out, header = args
    text = open(src).read()
    if "_m1_has(" in text or "def _m1_rounds(" not in text:
        raise SystemExit("IN must be a stable M1 file without M-B (layer_m1rt51s)")
    text = apply(text, DEVICE, "device")
    text = apply(text, [r if len(r) == 3 else (r[0], r[1], 1) for r in PY], "python")
    if "--round=3" in opts:
        text = apply(text, [(o, n, 1) for o, n in ROUND3], "round 3")
    a = text.index("def _m1_tune(state):")
    b = text.index("def _m1_enable(state):", a)
    body = text[a:b]
    if body.count(MODES_OLD) != 2:
        raise SystemExit("expected the two mode tuples in _m1_tune")
    text = text[:a] + body.replace(MODES_OLD, "M1_MODES") + text[b:]
    lines = text.split("\n")
    if lines[0] != "# okc: problem=muse_decode_layer":
        raise SystemExit("line 1 is not the okc header")
    lines[1] = header
    open(out, "w").write("\n".join(lines))
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
