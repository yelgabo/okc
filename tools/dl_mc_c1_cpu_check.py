"""CPU emulation of the attention boundary at one token: attn_boundary_kernel (8-CTA cluster of
256 threads) against M-C C-1's phase 1 sum of squares and phase 2 (tools/dl_mc_c1_patch.py,
mc1_kernel), both in FP32 with each kernel's summation order and the contractions the SASS shows
(s = fma(alpha, stream, beta * bf16r(f * rb)), the per-thread sums of squares by FMA, the scale
argument fma(total, 2^-13, eps)). rsqrtf is modelled as the FP32 rounding of the exact value in
both, so only the summation order differs, as on the GPU where both run the same MUFU.RSQ on
their totals. Also checks that C-1's row-pair and column partitions cover each item once for
every grid it accepts. FMA is emulated in FP64 then rounded to FP32 (a double rounding that can
differ from a true FMA in rare ties).

    python3 tools/dl_mc_c1_cpu_check.py [cases] [grid]
Needs torch (CPU); the okc-torch-headers image has it.
"""
import sys

import torch

H = 8192
HALF = H // 2
U = 8
GROUPS = H // 4
GMIN, GMAX = 64, 1024
EPS_POST, EPS_NEXT = 1e-6, 1e-6


def f32(x):
    return x.to(torch.float32)


def fma(a, b, c):
    return f32(a.double() * b.double() + c.double())


def bf16r(x):
    return x.to(torch.bfloat16).to(torch.float32)


def rsqrt(x):
    return f32(1.0 / torch.sqrt(x.double()))


def butterfly(v):
    """Lane-wise xor-shuffle sum over the last axis (32 lanes); every lane ends equal."""
    lanes = torch.arange(32)
    for o in (16, 8, 4, 2, 1):
        v = f32(v + v[..., lanes ^ o])
    return v[..., 0]


def seq_sum(v):
    """0 + v[0] + v[1] + ... in FP32, along the last axis."""
    acc = torch.zeros(v.shape[:-1], dtype=torch.float32)
    for i in range(v.shape[-1]):
        acc = f32(acc + v[..., i])
    return acc


def scale(total, eps):
    return rsqrt(fma(total, torch.tensor(2.0 ** -13), torch.tensor(eps, dtype=torch.float32)))


def thread_ss(x4):
    """A thread's chain over its 4 columns: ss = fma(x, x, ss) from zero."""
    ss = torch.zeros(x4.shape[:-1], dtype=torch.float32)
    for k in range(4):
        ss = fma(x4[..., k], x4[..., k], ss)
    return ss


def boundary_total(x):
    """attn_boundary_kernel's sum of squares of x [8192]: rank r, thread t owns columns
    r * 1024 + 4 t .. + 3; block_sum (warp butterfly, the 8 warp sums in lanes 0-7 of a second
    butterfly), then 0 + part[0] + ... + part[7]."""
    ss = thread_ss(x.view(8, 8, 32, 4))
    w = butterfly(ss)
    lanes = torch.cat([w, torch.zeros(8, 24)], dim=1)
    part = butterfly(lanes)
    return seq_sum(part[None])[0]


def mc1_total1(f, grid):
    """Phase 1: unit u of CTA b sums f0^2 then f1^2 by FMA over its pairs b U + u + i G U in
    order; the CTA sums its units in order; every CTA sums lane l's strided partials in order
    and butterflies."""
    pairs = f.view(HALF, 2)
    unit_ss = torch.zeros(grid * U, dtype=torch.float32)
    tu = grid * U
    for start in range(0, HALF, tu):
        c = torch.arange(start, min(start + tu, HALF))
        g = c - start
        unit_ss[g] = fma(pairs[c, 0], pairs[c, 0], unit_ss[g])
        unit_ss[g] = fma(pairs[c, 1], pairs[c, 1], unit_ss[g])
    part = seq_sum(unit_ss.view(grid, U))
    return grid_total(part)


def grid_total(part):
    grid = part.shape[0]
    pad = (-grid) % 32
    p = torch.cat([part, torch.zeros(pad)]).view(-1, 32).t()
    return butterfly(seq_sum(p)[None])[0]


def mc1_columns(grid):
    out = []
    for b in range(grid):
        g0 = b * GROUPS // grid
        out.append((g0, (b + 1) * GROUPS // grid - g0))
    return out


def boundary(stream, branch, alpha, beta, w_next):
    fa = branch.float()
    rb = scale(boundary_total(fa), EPS_POST)
    s = fma(alpha, stream, f32(beta * bf16r(f32(fa * rb))))
    rn = scale(boundary_total(s), EPS_NEXT)
    o = f32(f32(s * rn) * w_next.float())
    return o.to(torch.bfloat16), s, o


def mc1(stream, branch, alpha, beta, w_next, grid):
    fa = branch.float()
    rb = scale(mc1_total1(fa, grid), EPS_POST)
    s = fma(alpha, stream, f32(beta * bf16r(f32(fa * rb))))
    lane_ss = thread_ss(s.view(GROUPS, 4))
    part = torch.zeros(grid, dtype=torch.float32)
    for b, (g0, ng) in enumerate(mc1_columns(grid)):
        lanes = torch.zeros(32, dtype=torch.float32)
        lanes[:ng] = lane_ss[g0:g0 + ng]
        part[b] = butterfly(lanes[None])[0]
    rn = scale(grid_total(part), EPS_NEXT)
    o = f32(f32(s * rn) * w_next.float())
    return o.to(torch.bfloat16), s, o


def check_partitions():
    for grid in range(GMIN, GMAX + 1):
        cols = mc1_columns(grid)
        assert max(ng for _, ng in cols) <= 32, grid
        seen = torch.zeros(GROUPS, dtype=torch.int32)
        for g0, ng in cols:
            seen[g0:g0 + ng] += 1
        assert bool((seen == 1).all()), grid
        units = grid * U
        pairs = torch.zeros(HALF, dtype=torch.int32)
        for unit in range(units):
            pairs[unit::units] += 1
        assert bool((pairs == 1).all()), grid
    print(f"partitions: every grid {GMIN}..{GMAX} covers each row pair and float4 group once, "
          "at most 32 groups per CTA")


def main():
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 264
    check_partitions()
    gen = torch.Generator().manual_seed(31)
    stats = {"stream_flip_calls": 0, "stream_flips": 0, "next_flips": 0, "next_flip_max": 0,
             "rb_differs": 0, "worst_router_ulps_off_flips": 0.0, "worst_rel_all": 0.0,
             "worst_rel_off_flips": 0.0, "worst_flip_vs_bound": 0.0, "forced_flips": 0,
             "forced_flip_max": 0, "forced_classes_max": 0, "forced_worst_vs_bound": 0.0,
             "forced_rel": 0.0}
    for case in range(cases):
        xs = (0.05, 0.5, 2.0, 0.01)[case % 4]
        ss = (3.0, 1.0, 30.0, 0.3)[case // 4 % 4]
        branch = (torch.randn(H, generator=gen) * xs).to(torch.bfloat16)
        stream = torch.randn(H, generator=gen) * ss
        alpha = torch.rand(H, generator=gen) + 0.5
        beta = torch.rand(H, generator=gen) + 0.5
        w_next = (1 + 0.1 * torch.randn(H, generator=gen)).to(torch.bfloat16)
        n_ref, s_ref, o_ref = boundary(stream, branch, alpha, beta, w_next)
        n_got, s_got, o_got = mc1(stream, branch, alpha, beta, w_next, grid)
        fa = branch.float()
        if not torch.equal(scale(boundary_total(fa), EPS_POST),
                           scale(mc1_total1(fa, grid), EPS_POST)):
            stats["rb_differs"] += 1
        d = (s_got - s_ref).abs()
        flip = d > 0
        nflip = int(flip.sum())
        stats["stream_flips"] += nflip
        stats["stream_flip_calls"] += int(nflip > 0)
        if nflip:
            rb = float(torch.rsqrt(fa.double().pow(2).mean() + EPS_POST))
            ulp = torch.exp2(torch.floor(torch.log2(s_ref.abs().clamp(min=2.0 ** -126))) - 23)
            bound = beta * (fa * rb).abs() * 2.0 ** -7 + 8 * ulp
            stats["worst_flip_vs_bound"] = max(stats["worst_flip_vs_bound"],
                                               float((d[flip] / bound[flip]).max()))
        # The same step with the first scale moved one ulp up, which flips the BF16 rounding
        # of a few normalized branch values: the size the self-test's flip bound must admit.
        rb0 = scale(boundary_total(fa), EPS_POST)
        rb_up = torch.nextafter(rb0, torch.tensor(1e30))
        s_up = fma(alpha, stream, f32(beta * bf16r(f32(fa * rb_up))))
        du = (s_up - s_ref).abs()
        fu = du > 0
        stats["forced_flips"] += int(fu.sum())
        stats["forced_flip_max"] = max(stats["forced_flip_max"], int(fu.sum()))
        mant = branch.view(torch.int16)[fu] & 0x7F
        stats["forced_classes_max"] = max(stats["forced_classes_max"],
                                          int(torch.unique(mant).numel()))
        if fu.any():
            ulp = torch.exp2(torch.floor(torch.log2(s_ref.abs().clamp(min=2.0 ** -126))) - 23)
            bound = beta * (fa * rb0).abs() * 2.0 ** -7 + 8 * ulp
            stats["forced_worst_vs_bound"] = max(stats["forced_worst_vs_bound"],
                                                 float((du[fu] / bound[fu]).max()))
            stats["forced_rel"] = max(stats["forced_rel"], float(du.norm() / s_ref.norm()))
        keep = ~flip
        rd = (o_got - o_ref).abs()
        ulp_o = torch.exp2(torch.floor(torch.log2(o_ref.abs().clamp(min=2.0 ** -126))) - 23)
        stats["worst_router_ulps_off_flips"] = max(stats["worst_router_ulps_off_flips"],
                                                   float((rd[keep] / ulp_o[keep]).max()))
        stats["worst_rel_off_flips"] = max(stats["worst_rel_off_flips"],
                                           float(rd[keep].norm() / o_ref[keep].norm()))
        stats["worst_rel_all"] = max(stats["worst_rel_all"],
                                     float(rd.norm() / o_ref.norm()),
                                     float(d.norm() / s_ref.norm()))
        nb = n_got.view(torch.int16).int()
        rbits = n_ref.view(torch.int16).int()
        nd = (nb != rbits) & keep
        assert int((nb - rbits).abs()[nd].max()) <= 1 if nd.any() else True
        stats["next_flips"] += int(nd.sum())
        stats["next_flip_max"] = max(stats["next_flip_max"], int(nd.sum()))
    print(f"cases {cases} grid {grid}: " + ", ".join(
        f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
