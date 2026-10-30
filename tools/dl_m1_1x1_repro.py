"""Build the targeted reproduction of M1's 1x1 sliding gate failure (scored run d6c30f88c617438b:
one of 2000 M1-on replays off its reference by relative 3.2e-4 / 1.9e-4, element 2.2e-3 of the
row maximum, while the M1-off replays were bitwise) from the file that failed.

    python3 tools/dl_m1_1x1_repro.py muse_decode_layer/cand/layer_m1rt51.py OUT.py

The diagnostic replaces _m1_stress and _m1_tune; it never raises in the reproduction, leaves M1
off for the timed path and drops the prepare-time timeline (the tail keeps the SUMMARY). Per
shape (1x1 sliding with the gate's own seeds, then 1x8 sliding, where M1 does launch o and the
skf front as PDL secondaries):

  census   PDL launches per step that _m1_mode grants while each arm's graph is captured, and
           the kernel name sequence of one replay of kernel_fn's graph per arm (profiler): equal
           sequences with zero PDL launches mean the two arms run the same kernels
  arms     graph replays of kernel_fn's stages, M1 off / on / off2 / on2 on fresh inputs (the
           gate's five-entry bank, next entry per replay), off-kfn / on-kfn (kernel_fn itself,
           as the gate replays it, so only the next input and the stream are recorded), then
           off-same / on-same on one entry (the entry with most events in the fresh stage
           arms), then eager M1-off calls. Every replay
           writes one record on the device: per stage the relative L2 error against the eager
           M1-off reference of its input, the count of differing elements, and how many of those
           equal the reference of the previous replay's input (prev) or of an entry not read in
           the last two replays (oth); for the next attention input the largest element
           difference (index, observed, reference)
  events   per arm the replays where any stage differs, the stage where each first differs,
           per stage the number of replays with a difference, and up to M1R_DETAIL events in full

Reading: a stale or early read shows as a first-differing stage whose upstream stages are
bitwise and whose kernel is deterministic, in the fresh arms only; where the stage copies values
through, its differing elements also equal the previous input's reference (prev >> oth). A GEMV
of a partly stale x matches neither reference. Merge-order nondeterminism shows at the same rate in the same-input arms, as
one-ULP flips with prev and oth near zero, and at the same rate with M1 off and on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_m1_stage_diag as stage_diag  # noqa: E402

KFN = stage_diag.STAGES[stage_diag.STAGES.index("def _kfn_stages("):
                        stage_diag.STAGES.index("def _stage_acc(")]

REPRO = '''import math


M1R_STAGES = ("qkv", "attn", "o", "ffn", "xmoe", "tw", "moe", "unp", "hid", "str")
M1R_SHAPES = ((1, 1, "sliding", 4000), (1, 8, "sliding", 2000))
M1R_EAGER = 500
M1R_DETAIL = 4
_M1_CENSUS = []
_m1_mode_plain = _m1_mode


def _m1_mode(tokens, state):
    """_m1_mode, recording every grant: kernel_fn asks only where a PDL launch is possible (an
    sk o pick, an skf front), so the non-"off" answers count M1's PDL launches."""
    mode = _m1_mode_plain(tokens, state)
    _M1_CENSUS.append(mode)
    return mode


@@KFN@@
def _m1r_row(outs, ids, refs, ref_ids, i):
    """One replay's record (graph-capturable, device only)."""
    prev, oth = (i + 4) % 5, (i + 2) % 5
    vals = [i.float()[0]]
    for out, (r, norm, _) in zip(outs, refs):
        if out is None:
            vals += [torch.zeros_like(vals[0])] * 4
            continue
        o = out.float()
        ri = r.index_select(0, i)[0]
        d = o - ri
        diff = d != 0
        vals += [d.norm() / norm.index_select(0, i)[0], diff.sum().float(),
                 (diff & (o == r.index_select(0, prev)[0])).sum().float(),
                 (diff & (o == r.index_select(0, oth)[0])).sum().float()]
    vals.append((ids != ref_ids.index_select(0, i)[0]).sum().float())
    r, _, rmax = refs[8]
    ri = r.index_select(0, i)[0]
    d = (outs[8].float() - ri).reshape(-1)
    e = d.abs().argmax()
    cols = ri.shape[-1]
    vals += [e.float(), outs[8].reshape(-1)[e].float(), ri.reshape(-1)[e],
             rmax.index_select(0, i)[0].reshape(-1)[e // cols]]
    return torch.stack(vals)


def _m1r_events(rec, n):
    """Host summary of one arm: (line, detail lines, events per bank entry)."""
    ns = len(M1R_STAGES)
    rows = rec[:n].tolist()
    ev, first, per, prev, oth, ids, hid_rel, str_rel = [], {}, [0] * ns, 0, 0, 0, 0.0, 0.0
    by_i = [0] * 5
    for k, row in enumerate(rows):
        nnz = [row[1 + 4 * s + 1] for s in range(ns)]
        for s in range(ns):
            per[s] += int(nnz[s] > 0)
        ids += int(row[1 + 4 * ns] > 0)
        hid_rel = max(hid_rel, row[1 + 4 * 8])
        str_rel = max(str_rel, row[1 + 4 * 9])
        if not any(v > 0 for v in nnz):
            continue
        s0 = next(s for s in range(ns) if nnz[s] > 0)
        first[M1R_STAGES[s0]] = first.get(M1R_STAGES[s0], 0) + 1
        prev += int(row[1 + 4 * s0 + 2])
        oth += int(row[1 + 4 * s0 + 3])
        by_i[int(row[0])] += 1
        ev.append((k, s0, row))
    line = (f"n{n} events {len(ev)} first[" + ",".join(f"{s}:{c}" for s, c in first.items())
            + "] replays-differing[" + ",".join(f"{s}:{c}" for s, c in zip(M1R_STAGES, per) if c)
            + f"] prev/oth at first {prev}/{oth} ids {ids} rel hid {hid_rel:.1e} str "
            f"{str_rel:.1e} by-entry {by_i}")
    details = []
    for k, s0, row in ev[:M1R_DETAIL]:
        e, obs, ref, rmax = row[-4:]
        # BF16 keeps 8 significant bits, so one ULP at |ref| in [2^k, 2^(k+1)) is 2^(k-7).
        ulp = 2.0 ** (math.floor(math.log2(abs(ref))) - 7) if ref != 0 else None
        nz = ",".join(f"{M1R_STAGES[s]}:{int(row[1 + 4 * s + 1])}" for s in range(ns)
                      if row[1 + 4 * s + 1] > 0)
        details.append(
            f"r{k} i{int(row[0])} first {M1R_STAGES[s0]} rel {row[1 + 4 * s0]:.1e} "
            f"prev/oth {int(row[1 + 4 * s0 + 2])}/{int(row[1 + 4 * s0 + 3])} nnz[{nz}] hid "
            f"e{int(e)} obs {obs:.5g} ref {ref:.5g} el {abs(obs - ref) / max(rmax, 1e-12):.1e}"
            + (f" ulps {abs(obs - ref) / ulp:.1f}" if ulp else ""))
    return line, details, by_i


def _m1r_kernels(state, data, policy):
    """Kernel names of one replay of kernel_fn's graph under `policy`, in start order."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    saved = dict(state.get("m1_policy", {}))
    state["m1_policy"] = policy
    try:
        graph = _graph_build(lambda: kernel_fn(data, state))
    finally:
        state["m1_policy"] = saved
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    del graph
    evs = []
    for ev in prof.profiler.kineto_results.events():
        if ev.device_type() == DeviceType.CUDA:
            evs.append((_ev_ns(ev)[0], ev.name()))
    return [name for _, name in sorted(evs)]


def _m1_stress(state):
    """Diagnostic (never raises): see the docstring of tools/dl_m1_1x1_repro.py."""
    lines = []
    saved = dict(state.get("m1_policy", {}))
    on = {b: "o+front" for b in M1_BUCKETS}
    ns = len(M1R_STAGES)
    for batch, length, layer, n in M1R_SHAPES:
        tag = f"{batch}x{length}{layer[0]}"
        try:
            data = _timeline_case(state, batch, length, layer, 9300 + batch * length)
            dev = data["stream"].device
            gen = torch.Generator(device=dev).manual_seed(777 + batch * length)
            bank_s = torch.randn((5, batch * length, HIDDEN), generator=gen, device=dev)
            bank_a = (bank_s * torch.rsqrt(bank_s.pow(2).mean(-1, keepdim=True) + RMS_EPS)
                      ).to(torch.bfloat16)
            state["m1_policy"] = {}
            per, ids_l = [[] for _ in M1R_STAGES], []
            for k in range(5):
                data["stream"].copy_(bank_s[k])
                data["attn_input"].copy_(bank_a[k])
                outs, ids = _kfn_stages(data, state)
                for j, t in enumerate(outs):
                    per[j].append(t.clone())
                ids_l.append(ids.clone())
            refs = [_gate_refs(p) for p in per]
            ref_ids = torch.stack(ids_l)
            sel = torch.zeros(1, dtype=torch.long, device=dev)
            fixed = torch.zeros(1, dtype=torch.long, device=dev)
            pos = torch.zeros(1, dtype=torch.long, device=dev)
            rec = torch.zeros((n, 1 + 4 * ns + 1 + 4), device=dev)

            def step(fresh, whole=False):
                if fresh:
                    sel.add_(1)
                    i = sel % 5
                else:
                    i = fixed
                data["stream"].copy_(bank_s.index_select(0, i)[0])
                data["attn_input"].copy_(bank_a.index_select(0, i)[0])
                if whole:
                    # kernel_fn itself, as the gate replays it (no intermediates kept alive).
                    got = kernel_fn(data, state)
                    outs, ids = [None] * 8 + list(got[:2]), got[2]
                else:
                    outs, ids = _kfn_stages(data, state)
                rec.index_copy_(0, pos.clamp(max=n - 1), _m1r_row(outs, ids, refs, ref_ids, i)[None])
                pos.add_(1)

            try:
                k_off = _m1r_kernels(state, data, {})
                k_on = _m1r_kernels(state, data, on)
                same = k_off == k_on
                where = next((j for j, (a, b) in enumerate(zip(k_off, k_on)) if a != b),
                             min(len(k_off), len(k_on)))
                lines.append(f"{tag} kernels per replay off {len(k_off)} on {len(k_on)} "
                             + ("identical sequence" if same else
                                f"DIFFER at {where}: {k_off[where:where + 1]} vs "
                                f"{k_on[where:where + 1]}"[:240]))
            except Exception as exc:
                lines.append(f"{tag} kernels error {type(exc).__name__}: {str(exc)[:160]}")
            census = {}
            counts = [0] * 5
            arms = [("off", {}, True, False), ("on", on, True, False),
                    ("off2", {}, True, False), ("on2", on, True, False),
                    ("off-kfn", {}, True, True), ("on-kfn", on, True, True),
                    ("off-same", {}, False, False), ("on-same", on, False, False)]
            for arm, policy, fresh, whole in arms:
                if arm == "off-same":
                    fixed.fill_(max(range(5), key=lambda k: counts[k]))
                state["m1_policy"] = policy
                _M1_CENSUS.clear()
                try:
                    graph = _graph_build(lambda: step(fresh, whole))
                finally:
                    state["m1_policy"] = saved
                census[arm] = sum(m != "off" for m in _M1_CENSUS) / 3
                pos.zero_()
                rec.zero_()
                torch.cuda.synchronize()
                for _ in range(n):
                    graph.replay()
                torch.cuda.synchronize()
                del graph
                line, details, by_i = _m1r_events(rec, n)
                if fresh and not whole:
                    counts = [a + b for a, b in zip(counts, by_i)]
                same_i = f" entry {int(fixed)}" if not fresh else ""
                lines.append(f"{tag} {arm}{same_i} {line}")
                lines += [f"{tag} {arm} ev {d}" for d in details]
            state["m1_policy"] = {}
            pos.zero_()
            rec.zero_()
            m = min(M1R_EAGER, n)
            for _ in range(m):
                step(True)
            torch.cuda.synchronize()
            line, details, _ = _m1r_events(rec, m)
            lines.append(f"{tag} eager-off {line}")
            lines += [f"{tag} eager-off ev {d}" for d in details]
            lines.append(f"{tag} census pdl launches per step " + " ".join(
                f"{a} {c:g}" for a, c in census.items()))
        except Exception as exc:
            lines.append(f"{tag} error {type(exc).__name__}: {str(exc)[:200]}")
        finally:
            state["m1_policy"] = saved
    return lines


'''

TUNE = '''def _m1_tune(state):
    """Diagnostic build (tools/dl_m1_1x1_repro.py): the bit-16 self-test, then the
    reproduction; M1 stays off for the timed path."""
    state["m1_report"] = []
    rep = state["m1_report"]
    checked = _m1_selftest(state)
    rep.append(f"selftest PDL = plain bitwise on {len(checked)} launches")
    state["m1_ok"] = True
    try:
        rep.extend("repro " + line for line in _m1_stress(state))
    finally:
        state["m1_ok"] = False
        state["m1_policy"] = {}


'''


def main():
    src, out = sys.argv[1], sys.argv[2]
    text = open(src).read()
    a = text.index("def _m1_stress(state):")
    b = text.index("def _m1_pair(", a)
    text = text[:a] + REPRO.replace("@@KFN@@", KFN.rstrip("\n") + "\n\n") + text[b:]
    a = text.index("def _m1_tune(state):")
    b = text.index("def _m1_enable(state):", a)
    text = text[:a] + TUNE + text[b:]
    reps = [
        ("TIMELINE_SHAPES = (", "TIMELINE_SHAPES = () and (", 1),
        ("# okc: problem=muse_decode_layer\n",
         "# okc: problem=muse_decode_layer\n# M1 1x1 reproduction diagnostic (not a candidate)\n",
         1),
    ]
    for old, new, n in reps:
        if text.count(old) != n:
            raise SystemExit(f"expected {n} of {old[:60]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    open(out, "w").write(text)
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
