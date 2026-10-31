"""Bench-only copy of an M1 layer file whose prepare fits four times in the judge's 300 s, so a
four-arm bench of two M1 files can give every copy its own prepare (`--dup=own`) instead of
shared state, whose nulls proved too tight (a03039679e304d95). Two changes, both measurement
scaffolding and never part of a candidate:

- M1's policy is fixed (default {8: o+front, 16: o, 32: o}, what layer_m1rt51s picked in
  3aee22ba4a404c8a, c8d1e57261dd4901 and ad42a066bd3a4e01): _m1_tune still runs the bit-16
  self-test and the gate, then sets the policy without timing the chains and whole steps.
- The M1 gate replays REPLAYS times per arm and shape instead of 2000 (default 500), and so
  does the C-1 gate of an M-C file (MC1_STRESS_REPLAYS, --mc1-replays, default 500).

Apply the same arguments to both files of a bench so the arms differ only by the candidate.

    python3 tools/dl_bench_fix_m1.py IN.py OUT.py [--policy="{8: 'o+front', ...}"] [--replays=500]
"""
import ast
import sys

ANCHOR = '''    for line in _m1_stress(state):
        rep.append("gate " + line)
'''


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    if len(args) != 2 or not set(opts) <= {"policy", "replays", "mc1-replays"}:
        raise SystemExit(__doc__)
    policy = ast.literal_eval(opts.get("policy", "{8: 'o+front', 16: 'o', 32: 'o'}"))
    replays = int(opts.get("replays", "500"))
    # At 250 the M1 gate failed at 1x8 sliding in two of four copies (5ae9ce318e014edf: ids
    # flips and the element envelope against a control measured on too few calls), which
    # turned M1 off there; 500 passed in all four copies of b70cfa48c0264d20.
    if replays < 500 or int(opts.get("mc1-replays", "500")) < 500:
        raise SystemExit("gate replays below 500 break the M1 gate (5ae9ce318e014edf)")
    src, out = args
    text = open(src).read()
    if text.count(ANCHOR) != 1 or text.count("\nM1_STRESS_REPLAYS = 2000\n") != 1:
        raise SystemExit("IN is not an M1 file with the gate anchor and 2000 replays")
    text = text.replace(ANCHOR, ANCHOR + (
        f"    # Bench-only (tools/dl_bench_fix_m1.py): a fixed policy, so four prepares fit.\n"
        f"    state[\"m1_policy\"] = {policy!r}\n"
        f"    rep.append(\"bench: fixed policy {policy!r}, gate replays {replays}\")\n"
        f"    return\n"))
    text = text.replace("\nM1_STRESS_REPLAYS = 2000\n", f"\nM1_STRESS_REPLAYS = {replays}\n")
    if "\nMC1_STRESS_REPLAYS = 2000\n" in text:
        mc1 = int(opts.get("mc1-replays", "500"))
        text = text.replace("\nMC1_STRESS_REPLAYS = 2000\n", f"\nMC1_STRESS_REPLAYS = {mc1}\n")
    open(out, "w").write(text)
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
