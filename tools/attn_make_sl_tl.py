"""Generate the batch 16 / 32 sliding timeline and split check (CHECK ONLY, never a submission).

Timed shapes: decode 16 / 32 sliding (mode 1) and verify 16 / 32 sliding (mode 3) only.
Arms: ctrl (the judged source), ctrl2 (independent null), tl_probe (ONE_STAMP=3: per-CTA
globaltimer / clock stamps of the per-request path, attn_one.cu PR_BASE, at the control's
splits), tlA_probe (the same build at spA's splits) and spA / spB / spC (the control's module,
own state, other splits per request; host-only plan change). The stamp build compiles only
instances 1/1 and 8/3. ABCHECK keeps only the four timed shapes.

    python3 tools/attn_make_sl_tl.py OUT.py [split|nsr]
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

# (batch, q_len) -> splits per request. Control: 16 / 8 at decode 16 / 32, 8 / 4 at verify.
SPLITS = {
    "spA": {(16, 1): 32, (32, 1): 16, (16, 8): 16, (32, 8): 8},
    "spB": {(16, 1): 11, (32, 1): 11, (16, 8): 11, (32, 8): 6},
    "spC": {(16, 1): 8, (32, 1): 6, (16, 8): 6, (32, 8): 3},
}
SPLITS["tlA_probe"] = SPLITS["spA"]

# Variant "nsr" (check 2): ring depth of the per-request instances (ONE_NSR8 at verify,
# ONE_NSR1 at decode), the control's splits except n4s.
NSR = "-DONE_SL_ONLY=1"
VARIANTS = {
    "split": (["tl_probe@own=-DONE_STAMP=3,-DONE_SL_ONLY=1",
               "tlA_probe@own=-DONE_STAMP=3,-DONE_SL_ONLY=1",
               "spA@own=", "spB@own=", "spC@own="], SPLITS, "311401"),
    "nsr": (["tl_probe@own=-DONE_STAMP=3,-DONE_SL_ONLY=1",
             "n4@own=-DONE_NSR8=4,-DONE_NSR1=4," + NSR,
             "n5@own=-DONE_NSR8=5,-DONE_NSR1=5," + NSR,
             "n4s@own=-DONE_NSR8=4,-DONE_NSR1=4," + NSR,
             "tln5_probe@own=-DONE_STAMP=3,-DONE_NSR8=5,-DONE_NSR1=5," + NSR],
            {"n4s": {(16, 8): 16, (32, 8): 8}}, "311402"),
}

ONLY = ('#if ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)\n',
        '#if defined(ONE_SL_ONLY) && ONE_SL_ONLY\n#define ONE_INSTANCES(X) X(1, 1) X(8, 3)\n'
        '#elif ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)\n')

HELPERS = r'''
_SL_SPLITS = @@SPLITS@@
_PR_BASE, _PR_W, _PR_CLK = 1024 + 512 * 16 + 512 * 256, 96, 48


def _sl_splits(name, data):
    cfg = data["config"]
    if not cfg["window"]:
        return None
    return _SL_SPLITS.get(name, {}).get((cfg["batch"], cfg["q_len"]))


_SL_SUM = {}


def _sl_q(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return float("nan"), float("nan"), float("nan")
    return xs[0], xs[len(xs) // 2], xs[-1]


def _sl_stamps(reps=5):
    """Per-CTA timeline of the ONE_STAMP=3 arms (names starting with tl), cold calls.
    Times in us from the earliest CTA entry (globaltimer); per-CTA phase lengths from clock64
    at the median clock rate. Points: s0 entry, s4 page indices landed (after the barrier), s6
    prologue K/V issued, s8 query group landed, s13 query pass / new rows / patch done, t0..
    tile i ready, s16 loop end, s17 reds issued, s19 ticket (decode), s21 grid barrier passed
    and s22 finalize slice done (verify), s20 ticket path end (decode)."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for name in _AB_MODS:
        if not name.startswith("tl"):
            continue
        for batch, layer, q_len in _ab_scored():
            tag = f"b={batch:2d} {layer:7s} L={q_len}"
            data = _ab_shape_inputs(batch, layer, q_len, 1000)
            coop = q_len == 8
            for r in range(reps):
                with _ab_ctx(name):
                    n = _sl_splits(name, data) or _one_plan(
                        data["config"], data["page_table"].shape[1],
                        _one_state(data["qkv"].device)["resident"])[0]
                    buf = torch.zeros(_PR_BASE + n * batch * _PR_W, dtype=torch.int32,
                                      device="cuda")
                    flush.zero_()
                    _one(data, splits=n, dbg=buf)
                torch.cuda.synchronize()
                rec = buf[_PR_BASE:].view(n * batch, _PR_W).cpu().long()
                act = [i for i in range(n * batch) if int(rec[i, 13]) != 0]
                if not act:
                    print(f"TLSL {tag} {name} rep={r} no stamps", flush=True)
                    continue
                t0 = min(int(rec[i, 0]) for i in act)

                def gt(i, k):
                    return (((int(rec[i, k]) - t0 + (1 << 31)) % (1 << 32)) - (1 << 31)) / 1000.0

                def ck(i, a, b):
                    return ((int(rec[i, _PR_CLK + b]) - int(rec[i, _PR_CLK + a])) % (1 << 32))
                endk = 22 if coop else 20
                ghz = _sl_q([ck(i, 0, endk) / max(1.0, 1000.0 * (gt(i, endk) - gt(i, 0)))
                             for i in act])[1]

                def du(i, a, b):
                    return ck(i, a, b) / (ghz * 1000.0)
                ph = {
                    "entry": [gt(i, 0) for i in act],
                    "idx": [du(i, 0, 4) for i in act],
                    "kvissue": [du(i, 4, 6) for i in act],
                    "qgrp": [du(i, 6, 8) for i in act],
                    "prep": [du(i, 8, 13) for i in act],
                    "first": [du(i, 13, 24) for i in act],
                    "loop": [du(i, 24, 16) for i in act],
                    "red": [du(i, 16, 17) for i in act],
                    "loopend": [gt(i, 16) for i in act],
                    "end": [gt(i, endk) for i in act],
                }
                tiles = [int(rec[i, 45]) for i in act]
                gaps = []
                for i in act:
                    for k in range(1, min(16, int(rec[i, 45]))):
                        gaps.append(du(i, 24 + k - 1, 24 + k))
                ph["tilegap"] = gaps or [0.0]
                if coop:
                    ph["barrier"] = [du(i, 17, 21) for i in act]
                    ph["fin"] = [du(i, 21, 22) for i in act]
                    crit = max(act, key=lambda i: gt(i, 22))
                else:
                    ph["ticket"] = [du(i, 17, 19) for i in act]
                    fins = [i for i in act if int(rec[i, 46]) == 1]
                    ph["fin"] = [du(i, 19, 20) for i in fins] or [0.0]
                    crit = max(act, key=lambda i: gt(i, 20))
                span = max(ph["end"])
                ckeys = [0, 4, 6, 8, 13, 24, 16, 17] + ([21, 22] if coop else [19, 20])
                cline = " ".join(f"s{k}={gt(crit, k):.2f}" for k in ckeys)
                d = _SL_SUM.setdefault((tag, name), {})
                for k, v in ph.items():
                    lo, med, hi = _sl_q(v)
                    d.setdefault(k + "_med", []).append(med)
                    d.setdefault(k + "_max", []).append(hi)
                d.setdefault("span", []).append(span)
                for k in ckeys:
                    d.setdefault(f"crit_s{k}", []).append(gt(crit, k))
                print(f"TLSL {tag} {name} rep={r} ctas={n * batch} active={len(act)} "
                      f"ghz={ghz:.3f} span={span:.2f} tiles={_sl_q(tiles)} "
                      + " ".join("{}={:.2f}/{:.2f}/{:.2f}".format(k, *_sl_q(v))
                                 for k, v in ph.items())
                      + f" crit=c{crit}:sm{int(rec[crit, 44])}:t{int(rec[crit, 45])} {cline} "
                      f"{_ob_elapsed()}", flush=True)
            data = None
            torch.cuda.empty_cache()


def _sl_sum():
    for (tag, name), d in _SL_SUM.items():
        print(f"TLSUM {tag} {name} reps={len(d['span'])} " + " ".join(
            f"{k}={sorted(v)[len(v) // 2]:.2f}" for k, v in d.items()), flush=True)

'''


def main():
    out = Path(sys.argv[1])
    arms, splits, seed = VARIANTS[sys.argv[2] if len(sys.argv) > 2 else "split"]
    subprocess.run([sys.executable, str(ROOT / "tools/attn_make_bench_ab.py"), str(out),
                    "--null", "--shapes", "sliding", "--inputs", "3", "--rounds", "150",
                    "--seed", seed, *arms], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "tools/attn_inline_one.py"), str(out)],
                   check=True, cwd=ROOT)
    src = out.read_text()

    def sub(old, new):
        nonlocal src
        assert src.count(old) == 1, (old, src.count(old))
        src = src.replace(old, new)

    sub(ONLY[0], ONLY[1])
    first, rest = src.split("\n", 1)
    src = first + "\n# CHECK ONLY: batch 16 / 32 sliding timeline and split arms; never submit.\n" \
        + rest
    sub("        yield batch, layer, q_len",
        "        if batch == 1:\n            continue\n        yield batch, layer, q_len")
    sub("    shapes += [_OB_TESTS[i] for i in (0, 1, 2, 3, 7, 9, 10)]",
        "    # sliding timeline: the four batch 16 / 32 sliding shapes only.")
    sub("            return _one(data, mode=m) if m else _one(data)",
        "            if m:\n                return _one(data, mode=m)\n"
        "            n = _sl_splits(name, data)\n"
        "            return _one(data, splits=n) if n else _one(data)")
    sub("            return _ob_plan_str(data).replace(\" \", \",\")\n    except",
        "            n = _sl_splits(name, data)\n"
        "            return _ob_plan_str(data).replace(\" \", \",\") + "
        "(f\",splits_override={n}\" if n else \"\")\n    except")
    marker = "# Own-state arms: a private _ONE_STATE dict per arm, swapped in around each call."
    sub(marker, HELPERS.replace("@@SPLITS@@", repr(splits)) + marker)
    sub("        try:\n            _ab_bench()\n",
        "        try:\n            _sl_stamps()\n        except Exception as _exc:\n"
        "            print(f\"TLSL failed: {type(_exc).__name__}: {_exc}\", flush=True)\n"
        "        try:\n            _ab_bench()\n")
    sub("            _ab_stsum()\n", "            _ab_stsum()\n            _sl_sum()\n")
    out.write_text(src)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
