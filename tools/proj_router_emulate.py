"""CPU checks of the round-2 router index math in muse_bf16_proj/submission.py: rt_pack against
FP64 through the mma.sync fragment mapping, router_pack's box layout, and the wd XA push /
receive slot mapping (every cell of the 64 x 64 tile reaches its owner once per source rank).

    uv run --with torch python tools/proj_router_emulate.py [muse_bf16_proj/submission.py]
"""
import importlib.util
import sys

import torch


def check_packs(mod):
    torch.manual_seed(0)
    w = torch.randn(256, 8192, dtype=torch.float64)
    hi = w.to(torch.bfloat16)
    lo = (w - hi.double()).to(torch.bfloat16)
    pk = mod.rt_pack(hi, lo).view(16, 128, 8, 32, 8).double()
    x = torch.randn(3, 8192, dtype=torch.float64).to(torch.bfloat16).double()
    out = torch.zeros(3, 256, dtype=torch.float64)
    xs = x.view(3, 128, 4, 4, 4)  # token, k tile, t, q, 4
    for rb in range(16):
        for hl in range(2):
            for q in range(4):
                fr = pk[rb, :, hl * 4 + q].view(128, 8, 4, 8)  # k tile, g, t, 8 bf16
                xv = xs[:, :, :, q, :]
                # a0 / a2 pair with b0 / b1 for row g; a1 / a3 for row g + 8.
                rg = ((fr[None, :, :, :, 0:2] * xv[:, :, None, :, 0:2]).sum(-1)
                      + (fr[None, :, :, :, 4:6] * xv[:, :, None, :, 2:4]).sum(-1))
                rg8 = ((fr[None, :, :, :, 2:4] * xv[:, :, None, :, 0:2]).sum(-1)
                       + (fr[None, :, :, :, 6:8] * xv[:, :, None, :, 2:4]).sum(-1))
                out[:, rb * 16:rb * 16 + 8] += rg.sum((1, 3))
                out[:, rb * 16 + 8:rb * 16 + 16] += rg8.sum((1, 3))
    ref = x @ (hi.double() + lo.double()).T
    err = float((out - ref).abs().max())
    assert err < 1e-9, err
    hp = mod.router_pack(hi, lo).view(4, 128, 2, 64, 64)
    assert bool((hp[1, 5, 0] == hi[64:128, 320:384]).all() and (hp[2, 7, 1] == lo[128:192, 448:512]).all())
    print(f"rt_pack: max abs err {err:.1e}; router_pack: ok")


def check_xa():
    for clk in (2, 4, 8):
        tpr, nct = 64 // clk, 128
        xunits = nct * 8 // clk
        jw = tpr // 16 if tpr >= 16 else 1
        sent = {}
        for warp in range(4):
            for lane in range(32):
                r, q = 16 * warp + lane // 4, lane % 4
                cell = {}
                for j in range(8):
                    c = 8 * j + 2 * q
                    cell.update({4 * j: (r, c), 4 * j + 1: (r, c + 1), 4 * j + 2: (r + 8, c),
                                 4 * j + 3: (r + 8, c + 1)})
                for u in range(8):
                    if tpr >= 16:
                        dst, slot = warp * 16 // tpr, ((warp % jw) * 32 + lane) * 8 + u
                        v = [cell[4 * u + e] for e in range(4)]
                    else:
                        h, j0 = u >> 2, 2 * (u & 3)
                        dst, slot = 2 * warp + h, lane * 4 + (u & 3)
                        v = [cell[4 * j0 + 2 * h], cell[4 * j0 + 2 * h + 1], cell[4 * j0 + 4 + 2 * h],
                             cell[4 * j0 + 5 + 2 * h]]
                    assert 0 <= slot < xunits and (dst, slot) not in sent
                    sent[(dst, slot)] = v
                    assert all(dst * tpr <= t < (dst + 1) * tpr for t, _ in v)
        covered = set()
        for rk in range(clk):
            for it in range(xunits):
                if tpr >= 16:
                    lane = (it >> 3) & 31
                    t0 = 16 * (rk * jw + it // 256) + (lane >> 2)
                    t1, c0 = t0 + 8, 8 * (it & 7) + 2 * (lane & 3)
                    c1 = c0
                else:
                    lane = it >> 2
                    t0 = t1 = 16 * (rk >> 1) + 8 * (rk & 1) + (lane >> 2)
                    c0 = 16 * (it & 3) + 2 * (lane & 3)
                    c1 = c0 + 8
                got = [(t0, c0), (t0, c0 + 1), (t1, c1), (t1, c1 + 1)]
                assert sent[(rk, it)] == got, (clk, rk, it)
                covered.update(got)
        assert covered == {(t, c) for t in range(64) for c in range(64)}
        print(f"XA K split {clk}: {xunits} units per rank, every cell once")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "muse_bf16_proj/submission.py"
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    check_packs(mod)
    check_xa()


if __name__ == "__main__":
    main()
