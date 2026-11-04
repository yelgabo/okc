#!/usr/bin/env python3
"""Is the control's one-tile WS unit loop present, instruction for instruction, in a candidate's MID kernel?

    python3 tools/ly_ws2u_sass_cmp.py CONTROL.sass CAND.sass [kernel-substring]

Instructions are compared as text with addresses, encodings and branch-target labels stripped (registers,
predicates, immediates and shared-memory offsets kept). The control's unit loop is taken as the instruction run
from the first SYNCS.PHASECHK (ring-slot wait) before the first HMMA to the first BAR.SYNC after the last HMMA of
that HMMA cluster. Prints the loop's length, whether it occurs verbatim in the candidate, and the longest
candidate match otherwise.
"""
import difflib
import re
import sys

ADDR = re.compile(r"^\s*/\*[0-9a-f]+\*/\s*")
ENC = re.compile(r"/\*\s*0x[0-9a-f]+\s*\*/")
LAB = re.compile(r"`?\(?\.L_x_\d+\)?")


def kernels(path):
    out, cur = {}, None
    for line in open(path):
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur and re.match(r"^\s*/\*[0-9a-f]{4}\*/", line):
            t = ENC.sub("", ADDR.sub("", line)).strip().rstrip(";").strip()
            t = LAB.sub("L", t)
            if t and not t.startswith("NOP"):
                out[cur].append(t)
    return out


def pick(ks, sub):
    names = [k for k in ks if "moe_fused" in k and sub in k]
    return names


def loop_of(ins):
    hm = [i for i, t in enumerate(ins) if "HMMA" in t]
    if not hm:
        sys.exit("no HMMA")
    # first HMMA cluster: HMMAs within 64 instructions of each other
    first = hm[0]
    last = first
    for i in hm[1:]:
        if i - last > 64:
            break
        last = i
    s = max(i for i in range(first) if "SYNCS.PHASECHK" in ins[i] or "SYNCS.ARRIVE" in ins[i] and False) if any(
        "SYNCS.PHASECHK" in ins[i] for i in range(first)) else first
    e = next(i for i in range(last, len(ins)) if ins[i].startswith("BAR.SYNC") or " BAR.SYNC" in ins[i])
    return s, e + 1, sum(1 for i in hm if first <= i <= last)


def main():
    c, k = kernels(sys.argv[1]), kernels(sys.argv[2])
    sub = sys.argv[3] if len(sys.argv) > 3 else ""
    for name in pick(c, sub):
        if name not in k:
            print(name, "missing in candidate")
            continue
        ci, ki = c[name], k[name]
        s, e, nh = loop_of(ci)
        loop = ci[s:e]
        joined = "\n".join(ki)
        verb = ("\n".join(loop)) in joined
        sm = difflib.SequenceMatcher(None, loop, ki, autojunk=False)
        best = max(sm.get_matching_blocks(), key=lambda b: b.size)
        print(f"{name}: control {len(ci)} instr, candidate {len(ki)}; control loop [{s}, {e}) {len(loop)} instr, "
              f"{nh} HMMA; verbatim in candidate: {verb}; longest match {best.size} instr")
        if not verb:
            ops = sm.get_opcodes()
            shown = 0
            for tag, i1, i2, j1, j2 in ops:
                if tag != "equal" and shown < 12:
                    print("  ", tag, loop[i1:i2][:3], "->", ki[j1:j2][:3])
                    shown += 1


if __name__ == "__main__":
    main()
