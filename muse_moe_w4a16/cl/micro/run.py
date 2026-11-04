"""Build and run the CL-0.2 exchange micro-benchmarks on the box, holding the GPU lock only while the GPU runs.

    python muse_moe_w4a16/cl/micro/run.py [--replays 20] [--lock /root/okc/.build/box/gpu.lock]

Writes the raw log to .build/cl_micro/<tag>.log and prints the tables used in docs/runs-moe.md "CL-0.2 exchange".
"""
import argparse
import fcntl
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# w2 bytes per expert: 4096 x 4096 E2M1 codes plus one e4m3 scale per 16 values (TASK.md layout).
W2_BYTES = 4096 * 4096 * (0.5 + 1 / 16)
HBM_TBPS = 4.45  # box `sum` read rate, docs/runs-moe.md "T = 8 MoE floor and gap"


def build(out_dir):
    exe = os.path.join(out_dir, "exchange")
    src = os.path.join(HERE, "exchange.cu")
    cmd = ["nvcc", "-O3", "-std=c++17", "-arch=sm_90a", "-lineinfo", "-o", exe, src]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        sys.exit(r.returncode)
    r = subprocess.run(["cuobjdump", "-res-usage", exe], capture_output=True, text=True)
    return exe, r.stdout


def run_locked(exe, replays, lock):
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    print("compute apps before lock:", apps or "none", flush=True)
    with open(lock, "a") as fh:
        t = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)
        print(f"lock acquired after {time.time() - t:.1f} s", flush=True)
        try:
            r = subprocess.run([exe, str(replays)], capture_output=True, text=True)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        sys.exit(r.returncode)
    return r.stdout


def parse(text):
    rows = []
    for line in text.splitlines():
        if line.startswith(("XCH ", "LAT ")):
            d = {"tag": line[:3]}
            for k, v in re.findall(r"(\w+)=(\S+)", line):
                try:
                    d[k] = int(v)
                except ValueError:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            rows.append(d)
    return rows


def tables(rows):
    out = []
    x = [r for r in rows if r["tag"] == "XCH"]
    for r in x:
        r.setdefault("load", 0)
        r.setdefault("load_gbps", 0)
    empty = {(r["load"], r["T"], r["C"], r["ncl"]): r for r in x if r["mode"] == "empty"}
    out.append("| mode | load GB/s | T | C | clusters | chunk B | span med (min-max) | worst cluster med | "
               "to publication barrier | event med (min-max) | event net of empty | check |")
    out.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---|")
    for r in x:
        e = empty.get((r["load"], r["T"], r["C"], r["ncl"]))
        net = r["ev_med"] - e["ev_med"] if e else float("nan")
        ok = {1: "pass", 0: "FAIL", -1: "-"}[r["ok"]]
        lg = f"{r['load_gbps']:.0f}" if r["load"] else "-"
        out.append(f"| {r['mode']} | {lg} | {r['T']} | {r['C']} | {r['ncl']} | {r['chunk']} | "
                   f"{r['span_med']:.3f} ({r['span_min']:.3f}-{r['span_max']:.3f}) | {r['clmax_med']:.3f} | "
                   f"{r['pub_med']:.3f} | {r['ev_med']:.3f} ({r['ev_min']:.3f}-{r['ev_max']:.3f}) | {net:.3f} | {ok} |")
    out.append("")
    out.append("| latency | C / arrivals | clusters | median ns (min-max over replays) | p99 ns | other | SM GHz |")
    out.append("|---|---:|---:|---|---:|---|---:|")
    for r in rows:
        if r["tag"] != "LAT":
            continue
        if r["kind"] == "cluster_barrier":
            out.append(f"| cluster barrier round trip, arrive {r.get('sem', 'release')} | {r['C']} | {r['ncl']} | "
                       f"{r['med_ns']:.1f} ({r['med_min']:.1f}-{r['med_max']:.1f}) | {r['p99_ns']:.1f} | "
                       f"event {r['ev_med']:.2f} us for {r['iters']} iterations | {r['ghz']:.3f} |")
        else:
            out.append(f"| mbarrier handoff 8 warps to 1 | {r['arrivals']} | 1 CTA | {r['med_ns']:.1f} "
                       f"({r['med_min']:.1f}-{r['med_max']:.1f}) | {r['p99_ns']:.1f} | "
                       f"round trip {r['roundtrip_ns']:.1f} ns | {r['ghz']:.3f} |")
    out.append("")
    out.append("Gate arithmetic (worst-cluster span of the full grid; w2 stream per slice = 9.44 MB / S at "
               f"{HBM_TBPS} TB/s, and the per-cluster time with every co-resident cluster sharing HBM):")
    out.append("| mode | load | T | C | S | exchange us | publication us | w2 per slice, whole-GPU rate | ratio | "
               "w2 per slice, concurrent clusters | ratio | publication ratio |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    w2_us = W2_BYTES / (HBM_TBPS * 1e12) * 1e6
    for r in x:
        if r["ncl"] <= 1 or r["mode"] == "empty":
            continue
        s = 16 // r["C"]
        lone = w2_us / s
        conc = lone * r["ncl"]
        ex = r["clmax_med"]
        out.append(f"| {r['mode']} | {r['load']} | {r['T']} | {r['C']} | {s} | {ex:.3f} | {r['pub_med']:.3f} | "
                   f"{lone:.3f} | {ex / lone:.2f} | {conc:.2f} | {ex / conc:.3f} | {r['pub_med'] / conc:.3f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--lock", default="/root/okc/.build/box/gpu.lock")
    ap.add_argument("--tag", default=time.strftime("xch_%Y%m%d_%H%M%S"))
    ap.add_argument("--parse", help="re-print the tables of an existing log")
    a = ap.parse_args()
    if a.parse:
        print(tables(parse(open(a.parse).read())))
        return
    out_dir = os.path.join(ROOT, ".build", "cl_micro")
    os.makedirs(out_dir, exist_ok=True)
    exe, res = build(out_dir)
    text = run_locked(exe, a.replays, a.lock)
    log = os.path.join(out_dir, a.tag + ".log")
    with open(log, "w") as fh:
        fh.write(res + "\n" + text)
    print(text)
    print(tables(parse(text)))
    print("log:", log)


if __name__ == "__main__":
    main()
