"""M-B slice 1 on a stable M1 file (tools/dl_stable_tuner_patch.py --m1 output, e.g.
layer_m1rt51s.py): a fourth M1 mode, "o+front+unp", that also launches the unpooler projection
as a PDL secondary of the MoE when its pick is a single-CTA sk config, so its first ring of
weights streams while the MoE drains. Python only: the sk kernels built with SK_PDL already
take bit 16 (M1's o path), so no device code changes.

Idea learned from the F-LAYER-0 comparison (four-arm checks 20abe6291eb04ed6,
00607fc6f8c84c77) of alygg77/okc-muse-spark-kernels @ 6cae49a, whose layer chains its glue and
projection kernels by PDL at T <= 32 with weights issued before the wait. Re-implemented on
this layer's own M1 machinery; no donor code.

What changes:
- _m1_pdl(layer, tokens, state) decides the PDL bit in _proj: o in every M1 mode but off, the
  unpooler only in "o+front+unp"; the two front call sites accept "o+front+unp" as "o+front".
- _m1_selftest also launches every sk unpooler pick at 1-32 tokens with the bit, bitwise
  against the plain launch.
- _m1_stress gates M1_GATE_MODE ("o+front+unp", the superset of launches) instead of
  "o+front"; bounds, census and negative controls unchanged.
- _m1_tune times the new mode beside off / o / o+front; _stable_pick keeps the preferred mode
  unless the new one saves more than the noise in 8 of 9 rounds, so its paired whole-step
  saving is printed (SUMMARY "stable m1") whether or not it is picked.

    python3 tools/dl_mb_unp_patch.py IN.py OUT.py "HEADER LINE 2"
"""
import sys

MODES_OLD = '("off", "o", "o+front")'
MODES_NEW = '("off", "o", "o+front", "o+front+unp")'

REPS = [
    ('''            if layer == "o" and getattr(pick[1], "sk", False) and _m1_mode(m, state) != "off":
                return pick[1](x.contiguous(), True)''',
     '''            if getattr(pick[1], "sk", False) and _m1_pdl(layer, m, state):
                return pick[1](x.contiguous(), True)'''),
    ('''    """M1's choice for this token count: "off", "o" or "o+front"."""''',
     '''    """M1's choice for this token count: "off", "o", "o+front" or "o+front+unp"."""'''),
    ('''M1_STRESS_REPLAYS = 2000
''',
     '''def _m1_pdl(layer, tokens, state):
    """Whether this projection launches as a PDL secondary: o under every M1 mode but off,
    the unpooler (whose predecessor is the MoE) only under "o+front+unp"."""
    mode = _m1_mode(tokens, state)
    if layer == "o":
        return mode != "off"
    return layer == "unpooler" and mode == "o+front+unp"


def _m1_front(tokens, state):
    return _m1_mode(tokens, state) in ("o+front", "o+front+unp")


# The policy M1's gate replays: the mode with the most PDL launches, so every launch any picked
# mode makes is covered.
M1_GATE_MODE = "o+front+unp"
M1_STRESS_REPLAYS = 2000
'''),
    ('''    on = {b: "o+front" for b in M1_BUCKETS}''',
     '''    on = {b: M1_GATE_MODE for b in M1_BUCKETS}'''),
    ('''            checked.append(f"skf{bucket_of(m)}@{m}")
    return checked''',
     '''            checked.append(f"skf{bucket_of(m)}@{m}")
        pick = state["skp_pick"].get(("unpooler", m))
        if pick is not None and getattr(pick[1], "sk", False):
            x = torch.randn((m, state["dense"]["unpooler"].shape[1]), generator=gen, device=dev)
            x = x.to(torch.bfloat16)
            a, b = pick[1](x, False), pick[1](x, True)
            torch.cuda.synchronize()
            if not torch.equal(a, b):
                raise RuntimeError(f"unpooler {pick[0]} m={m}: PDL launch differs")
            checked.append(f"unp:{pick[0]}@{m}")
    return checked'''),
    ('''    every fused pooler + router config, launched as a PDL secondary, against the plain launch on''',
     '''    every fused pooler + router config and every sk unpooler pick, launched as a PDL secondary,
    against the plain launch on'''),
]


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, out, header = sys.argv[1:]
    text = open(src).read()
    if "_m1_pdl(" in text:
        raise SystemExit("IN already has the slice")
    if "def _m1_rounds(" not in text:
        raise SystemExit("IN is not a stable M1 file (tools/dl_stable_tuner_patch.py --m1)")
    for old, new in REPS:
        if text.count(old) != 1:
            raise SystemExit(f"not where expected: {old[:70]!r}")
        text = text.replace(old, new)
    front = '_m1_mode(tokens, state) == "o+front")'
    if text.count(front) != 2:
        raise SystemExit("expected the two front call sites")
    text = text.replace(front, "_m1_front(tokens, state))")
    a = text.index("def _m1_tune(state):")
    b = text.index("def _m1_enable(state):", a)
    body = text[a:b]
    if body.count(MODES_OLD) != 2:
        raise SystemExit("expected the two mode tuples in _m1_tune")
    text = text[:a] + body.replace(MODES_OLD, MODES_NEW) + text[b:]
    lines = text.split("\n")
    if lines[0] != "# okc: problem=muse_decode_layer":
        raise SystemExit("line 1 is not the okc header")
    lines[1] = header
    open(out, "w").write("\n".join(lines))
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
