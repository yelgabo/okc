"""CL-0.6 epilogue-warp bench: consumer stall per tile, form A (epilogue warp) against form B (consumers store, bar.sync
and release), with form C (compute only) as the baseline.

  python muse_moe_w4a16/cl/micro/epilogue_bench.py --lock /root/okc/.build/box/gpu.lock
  python muse_moe_w4a16/cl/micro/epilogue_bench.py --sass-only

Each config is one kernel inside a CUDA graph; every replay follows a 1 GiB L2 flush, forms are interleaved per replay.
Stall per tile, two estimators per form X:
  span:  (median consumer-warp loop span of X - same for C) / ntiles, from stamps-off builds (loop start / end only);
  stamp: mean over tiles of (t after handoff - t after compute) from stamps-on builds, minus the same for C.
"""
import argparse
import ctypes
import fcntl
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "epilogue_warp.cu")
BUILD = os.path.join(os.path.expanduser("~"), ".cache", "okc_cl_micro")
FORMS = {0: "A", 1: "B", 2: "C"}
NT_MAX = 64
PATS = dict(HMMA=r"\bHMMA", STG=r"\bSTG", STS=r"\bSTS", MEMBAR=r"\bMEMBAR", BAR=r"\bBAR\b|\bBAR\.", SYNCS=r"\bSYNCS",
            RED=r"\bRED\b|\bREDG|\bRED\.", ATOM=r"\bATOM", ERRBAR=r"ERRBAR", CCTL=r"\bCCTL", WARPSYNC=r"WARPSYNC")


def build():
    os.makedirs(BUILD, exist_ok=True)
    so = os.path.join(BUILD, "epilogue_warp.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(SRC):
        subprocess.check_call(["nvcc", "-O3", "-std=c++17", "-arch=sm_90a", "-shared", "-Xcompiler", "-fPIC",
                               "-lineinfo", "-o", so, SRC])
    return so


def sass_report(so):
    txt = subprocess.check_output(["cuobjdump", "-sass", so], text=True)
    funcs = re.split(r"\n\s*Function : ", txt)[1:]
    rows = []
    for f in funcs:
        name = f.split("\n", 1)[0].strip()
        m = re.search(r"epi_kernelILi(\d)ELi(\d+)ELb(\d)ELb(\d)E", name)
        if not m:
            continue
        form, T, load, stamps = (int(x) for x in m.groups())
        ins = []
        for line in f.split("\n"):
            mm = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
            if mm:
                ins.append((int(mm.group(1), 16), mm.group(2).strip()))
        # the consumer tile loop is the widest conditional backward branch whose body holds the HMMA chain
        loops, epi = [], []
        for addr, op in ins:
            # conditional only: unconditional backward BRAs return from out-of-line slow paths (mbarrier spin)
            bm = re.search(r"^@!?U?P\w+\s+BRA\b.*?(0x[0-9a-f]+)", op)
            if bm:
                tgt = int(bm.group(1), 16)
                if tgt < addr:
                    body = [o for a, o in ins if tgt <= a <= addr]
                    if any("HMMA" in o for o in body):
                        loops.append((tgt, addr, body))
                    elif form == 0 and any("LDS" in o for o in body) and any("STG" in o for o in body):
                        epi.append((tgt, addr, body))
        if not loops:
            continue
        tile = max(loops, key=lambda l: l[1] - l[0])
        body = tile[2]

        def cnt(pat):
            return sum(1 for o in body if re.search(pat, o))

        if epi:
            e = max(epi, key=lambda l: l[1] - l[0])[2]
            rows.append(dict(form="A-epilogue-warp", T=T, load=load, stamps=stamps, loop_ins=len(e), total_ins=len(ins),
                             **{k: sum(1 for o in e if re.search(v, o)) for k, v in PATS.items()},
                             membar_lines=[o for o in e if re.search(r"MEMBAR|\bBAR|SYNCS|RED|ERRBAR|CCTL", o)]))
        rows.append(dict(form=FORMS[form], T=T, load=load, stamps=stamps, loop_ins=len(body), total_ins=len(ins),
                         HMMA=cnt(r"\bHMMA"), STG=cnt(r"\bSTG"), STS=cnt(r"\bSTS"), MEMBAR=cnt(r"\bMEMBAR"),
                         BAR=cnt(r"\bBAR\b|\bBAR\."), SYNCS=cnt(r"\bSYNCS"), RED=cnt(r"\bRED\b|\bREDG|\bRED\."),
                         ATOM=cnt(r"\bATOM"), ERRBAR=cnt(r"ERRBAR"), CCTL=cnt(r"\bCCTL"), WARPSYNC=cnt(r"WARPSYNC"),
                         membar_lines=[o for o in body if re.search(r"MEMBAR|\bBAR|SYNCS|RED|ERRBAR|CCTL", o)]))
    return sorted(rows, key=lambda r: (r["stamps"], r["load"], r["T"], r["form"]))


class Lock:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666) if path else None

    def __enter__(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_EX)

    def __exit__(self, *a):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)


def med_spread(xs):
    xs = sorted(xs)
    return statistics.median(xs), xs[0], xs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default=None)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--ntiles", type=int, default=64)
    ap.add_argument("--grid", type=int, default=132)
    ap.add_argument("--kiters", default="4,32")
    ap.add_argument("--T", default="8,32")
    ap.add_argument("--load", default="0,1")
    ap.add_argument("--sass-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    so = build()
    sass = sass_report(so)
    print("SASS (consumer tile loop of each instantiation)")
    print("form T load stamps loop_ins total_ins HMMA STS STG MEMBAR BAR SYNCS RED ATOM WARPSYNC")
    for r in sass:
        print(f"SASS {r['form']} {r['T']:>2} {r['load']} {r['stamps']} {r['loop_ins']:>4} {r['total_ins']:>5} "
              f"{r['HMMA']:>3} {r['STS']:>3} {r['STG']:>3} {r['MEMBAR']:>3} {r['BAR']:>3} {r['SYNCS']:>3} "
              f"{r['RED']:>3} {r['ATOM']:>3} {r['WARPSYNC']:>3}")
        if not r["stamps"] and not r["load"]:
            for o in r["membar_lines"]:
                print(f"SASSLINE {r['form']} T={r['T']}  {o}")
    if a.sass_only:
        return

    import torch
    lib = ctypes.CDLL(so)
    lib.epi_launch.restype = ctypes.c_int
    lib.epi_launch.argtypes = [ctypes.c_int] * 7 + [ctypes.c_void_p] * 4 + [ctypes.c_ulonglong] + [ctypes.c_void_p] * 4
    dev = torch.device("cuda")
    G, NT = a.grid, a.ntiles
    seed = torch.randint(0, 2**31 - 1, (64,), dtype=torch.int32, device=dev)
    out = torch.empty(G * NT * 8 * 16 * 32 // 2, dtype=torch.int32, device=dev)
    cnt = torch.zeros(G * 2, dtype=torch.int32, device=dev)
    region = ((1 << 30) // G) // 32768 * 32768
    wbuf = torch.empty(G * region, dtype=torch.uint8, device=dev)
    wbuf.random_(0, 255)
    flush = torch.empty(1 << 28, dtype=torch.int32, device=dev)  # 1 GiB
    flush.fill_(1)
    stream = torch.cuda.Stream()

    results = []
    for T in [int(x) for x in a.T.split(",")]:
        for load in [int(x) for x in a.load.split(",")]:
            for kit in [int(x) for x in a.kiters.split(",")]:
                for stamps in (0, 1):
                    graphs = {}
                    bufs = {}
                    for form in (0, 1, 2):
                        st = torch.zeros(G * 8 * NT_MAX * 2, dtype=torch.int32, device=dev)
                        sp = torch.zeros(G * 20, dtype=torch.int64, device=dev)
                        ch = torch.zeros(G, dtype=torch.int64, device=dev)
                        args = (form, T, load, stamps, G, NT, kit, seed.data_ptr(), out.data_ptr(), cnt.data_ptr(),
                                wbuf.data_ptr(), region, st.data_ptr(), sp.data_ptr(), ch.data_ptr())
                        with Lock(a.lock):
                            with torch.cuda.stream(stream):
                                rc = lib.epi_launch(*args, ctypes.c_void_p(stream.cuda_stream))
                                assert rc == 0, rc
                            stream.synchronize()
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g, stream=stream):
                                rc = lib.epi_launch(*args, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
                                assert rc == 0, rc
                            torch.cuda.synchronize()
                        graphs[form], bufs[form] = g, (st, sp, ch)
                    rec = {f: dict(ev=[], span=[], cons=[], stall=[], epi=[], chunks=[]) for f in (0, 1, 2)}
                    with Lock(a.lock):
                        for rep in range(a.replays + 2):
                            for k in range(3):
                                form = (rep + k) % 3
                                flush.add_(1)
                                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                e0.record()
                                graphs[form].replay()
                                e1.record()
                                torch.cuda.synchronize()
                                if rep < 2:
                                    continue
                                st, sp, ch = bufs[form]
                                spv = sp.view(G, 20).cpu()
                                starts = spv[:, 0:16:2]
                                ends = spv[:, 1:16:2]
                                kend = spv[:, 17].max().item() if form == 0 else 0
                                r = rec[form]
                                r["ev"].append(e0.elapsed_time(e1) * 1e3)
                                r["span"].append((max(ends.max().item(), kend) - starts.min().item()) / 1e3)
                                r["cons"].append(float((ends - starts).double().median().item()))
                                if load:
                                    r["chunks"].append(float(ch.double().mean().item()))
                                if stamps:
                                    s = st.view(G, 8, NT_MAX, 2)[:, :, :NT, :].cpu().long()
                                    r["stall"].append(float((s[..., 1] - s[..., 0]).double().mean().item()))
                    for form in (0, 1, 2):
                        r = rec[form]
                        row = dict(form=FORMS[form], T=T, load=load, kiters=kit, stamps=stamps,
                                   ev_us=med_spread(r["ev"]), span_us=med_spread(r["span"]),
                                   cons_ns=med_spread(r["cons"]))
                        if r["stall"]:
                            row["stall_ns"] = med_spread(r["stall"])
                        if r["chunks"]:
                            row["chunks"] = statistics.median(r["chunks"])
                        results.append(row)
                        def f3(t):
                            return f"{t[0]:9.1f} [{t[1]:.1f}, {t[2]:.1f}]"
                        line = (f"RUN T={T:>2} load={load} kit={kit:>2} stamps={stamps} form={FORMS[form]} "
                                f"event_us {f3(row['ev_us'])} span_us {f3(row['span_us'])} "
                                f"cons_loop_ns {f3(row['cons_ns'])}")
                        if "stall_ns" in row:
                            line += f" stamp_handoff_ns {f3(row['stall_ns'])}"
                        print(line, flush=True)
    print("SUMMARY stall per tile (ns), span estimator = (cons_loop X - cons_loop C) / ntiles; stamp estimator = "
          "handoff X - handoff C")
    key = {(r["form"], r["T"], r["load"], r["kiters"], r["stamps"]): r for r in results}
    for T in [int(x) for x in a.T.split(",")]:
        for load in [int(x) for x in a.load.split(",")]:
            for kit in [int(x) for x in a.kiters.split(",")]:
                c0 = key[("C", T, load, kit, 0)]["cons_ns"][0]
                c1 = key[("C", T, load, kit, 1)]["stall_ns"][0]
                parts = []
                for f in ("A", "B"):
                    s0 = (key[(f, T, load, kit, 0)]["cons_ns"][0] - c0) / NT
                    s1 = key[(f, T, load, kit, 1)]["stall_ns"][0] - c1
                    parts.append(f"{f}: span {s0:7.1f} stamp {s1:7.1f}")
                ctile = c0 / NT
                print(f"SUMMARY T={T:>2} load={load} kit={kit:>2} compute/tile(C) {ctile:7.1f}  " + "  ".join(parts) +
                      f"  kernel span A/B/C us {key[('A', T, load, kit, 0)]['span_us'][0]:.2f} / "
                      f"{key[('B', T, load, kit, 0)]['span_us'][0]:.2f} / {key[('C', T, load, kit, 0)]['span_us'][0]:.2f}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(sass=sass, results=results), fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
