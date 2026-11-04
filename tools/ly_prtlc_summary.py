#!/usr/bin/env python3
"""Consumer-side stage clocks from a tools/ly_box_prtl.py JSON of a dl_ly_prtlc_patch.py build (--raw N).

    python3 tools/ly_prtlc_summary.py tl.json [cell-substring]

Per raw input: for every stage s, consumer warp 0's wait (its full-barrier return minus its done stamp of stage
s - 1) and its processing time (done minus return), averaged over CTAs; the sums over pooler and router stages;
and the median CTA's clocks. Router stages at T <= 8 under GLE_ILV 1 are s % 5 == 4.
"""
import json
import statistics as st
import sys


def main(path, sub=""):
    d = json.load(open(path))
    for row in d["rows"]:
        lab = f"{row['case']['load']} {row['case']['layer']}"
        if sub not in lab:
            continue
        T = row["T"]
        for inp in row["inputs"]:
            if "raw" not in inp:
                continue
            v = inp["raw"]
            t0 = min(r[0] for r in v)
            u = lambda x: (x - t0) / 1e3  # noqa: E731
            n = 20
            waits = [[] for _ in range(n)]
            procs = [[] for _ in range(n)]
            ret0_after_wait, le_after_done19 = [], []
            for r in v:
                ret = [u(r[36 + s]) for s in range(n)]
                done = [u(r[16 + s]) for s in range(n)]
                for s in range(n):
                    prev = done[s - 1] if s else u(r[3])
                    waits[s].append(ret[s] - prev)
                    procs[s].append(done[s] - ret[s])
                ret0_after_wait.append(ret[0] - u(r[3]))
                le_after_done19.append(max(u(r[56 + w]) for w in range(8)) - done[19])
            isr = lambda s: (s % 5 == 4) if T <= 8 else s >= 16  # noqa: E731
            mw = [st.mean(x) for x in waits]
            mp = [st.mean(x) for x in procs]
            print(f"== {lab} T {T} seed {inp['seed']} post_wait {inp['post_wait']:.2f}")
            print("  stage:   " + " ".join(f"{s:5d}" for s in range(n)))
            print("  wait:    " + " ".join(f"{x:5.2f}" for x in mw))
            print("  process: " + " ".join(f"{x:5.2f}" for x in mp))
            pw = sum(x for s, x in enumerate(mw) if not isr(s))
            pp = sum(x for s, x in enumerate(mp) if not isr(s))
            rw = sum(x for s, x in enumerate(mw) if isr(s))
            rp = sum(x for s, x in enumerate(mp) if isr(s))
            print(f"  pooler stages: wait {pw:.2f} process {pp:.2f} (per stage {pp / 16:.3f}); router stages: wait {rw:.2f} process {rp:.2f} (per stage {rp / 4:.3f});"
                  f" first return after wait {st.mean(ret0_after_wait):.2f}; loop exit after done19 {st.mean(le_after_done19):.2f}")
            tot = pw + pp + rw + rp
            print(f"  wait total {pw + rw:.2f} ({(pw + rw) / tot * 100:.0f}%), process total {pp + rp:.2f} ({(pp + rp) / tot * 100:.0f}%), loop {tot:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
