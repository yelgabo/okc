"""Per shape, the MoE expert kernels' GPU time from okc run JSONs on muse_decode_layer: W4
GEMM1, its split-K SwiGLU reduce, W4 GEMM2, the W4 finalizer, and Marlin (both GEMMs share
one row), against an earlier run's Marlin time. A kernel outside a report's listed rows
("... N more") shows as '-'.

    python3 tools/dl_moe_split.py muse_decode_layer/profiles/submission.<job>.json [<earlier>.json]

Profiler times include time a kernel spends in its programmatic-dependent-launch wait.
"""

import json
import re
import sys

PARTS = (
    ("g1", r"w4_gemm_kernel<1,"),
    ("swiglu", r"w4_swiglu_kernel"),
    ("g2", r"w4_gemm_kernel<2,"),
    ("fin", r"w4_finalize_kernel"),
    ("marlin", r"marlin_moe::Marlin"),
)


def load(path):
    d = json.load(open(path))
    return d.get("result", d) if "benchmarks" not in d else d


def split(report):
    out = {}
    for line in report.splitlines()[1:]:
        m = re.match(r"(.+?)\s+([\d.]+)\s*µs\s+([\d.]+)%\s+(\d+)\s*$", line)
        if not m:
            continue
        for key, pat in PARTS:
            if re.search(pat, m.group(1)):
                out[key] = out.get(key, 0.0) + float(m.group(2))
    return out


def by_shape(run):
    return {(p["case"]["load"], p["case"]["layer"]): split(p["report"])
            for p in run.get("profiles", [])}


def main():
    run = by_shape(load(sys.argv[1]))
    prev = by_shape(load(sys.argv[2])) if len(sys.argv) > 2 else {}
    keys = [k for k, _ in PARTS]
    print("| shape | " + " | ".join(f"{k} us" for k in keys)
          + " | W4 total us | earlier Marlin us |")
    print("|---" * (len(keys) + 3) + "|")
    for shape, t in run.items():
        w4 = sum(t.get(k, 0.0) for k in ("g1", "swiglu", "g2", "fin"))
        before = prev.get(shape, {}).get("marlin")
        cells = [f"{t[k]:.1f}" if k in t else "-" for k in keys]
        tail = f"{before:.1f}" if before is not None else ""
        print(f"| {shape[0]} {shape[1]} | " + " | ".join(cells) + f" | {w4:.1f} | {tail} |")


if __name__ == "__main__":
    main()
