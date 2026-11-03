"""CPU check of ra_kernel's operand path (muse_bf16_proj/experiments/ra_wgmma.py): ra_pack's
stream and escape lists, decoded per lane exactly as the kernel does (decode8, XOR patch), must
give every weight's exact BF16 bits at its m64nNk16 register-A position (the mma.m16n8k16 A
layout per warp: a0 row g k 2a..2a+1, a1 row g+8, a2 row g k 2a+8..+9, a3 row g+8).
  uv run --with torch --with numpy python tools/proj_ra_emulate.py"""
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("ra", "muse_bf16_proj/experiments/ra_wgmma.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def prmt(a, b, sel):
    by = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    return sum(by[(sel >> (4 * i)) & 7] << (8 * i) for i in range(4))


def decode8(lw0, lw1, codes, base4):
    ha = (((codes & 0x07070707) + base4) | ((codes << 4) & 0x80808080)) & 0xFFFFFFFF
    hb = ((((codes >> 4) & 0x07070707) + base4) | (codes & 0x80808080)) & 0xFFFFFFFF
    return [prmt(lw0, ha, 0x5140), prmt(lw0, ha, 0x7362), prmt(lw1, hb, 0x5140), prmt(lw1, hb, 0x7362)]


def check(n, clk, ku, seed):
    k = 64 * clk * ku
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02)
    # Escapes: a few huge and tiny weights per warp slice, plus zeros.
    idx = torch.randint(n * k, (n * k // 2000,), generator=g)
    w.view(-1)[idx[: len(idx) // 2]] *= 2.0 ** 12
    w.view(-1)[idx[len(idx) // 2:]] *= 2.0 ** -14
    w.view(-1)[torch.randint(n * k, (50,), generator=g)] = 0.0
    w = w.to(torch.bfloat16)
    pk = mod.rd_pack(w[:, mod.ra_perm(k)].contiguous(), (1, 8, clk, ku, 0))
    assert pk is not None, "escape overflow"
    st = pk["stream"].numpy().tobytes()
    esc = pk["esc"].numpy().astype("uint32")
    base4 = pk["base4"]
    bits = w.view(torch.int16).numpy().astype("uint16")
    nrg = n // 16
    bad = 0
    for rg in range(nrg):
        for rk in range(clk):
            lst = [int(v) for v in esc[rg * clk + rk]]
            for s in range(ku):
                u = rk * ku + s
                off = (u * nrg + rg) * 1536
                for lane in range(32):
                    gq, a = lane >> 2, lane & 3
                    word = lambda o: [int.from_bytes(st[off + o + lane * 16 + 4 * i: off + o + lane * 16 + 4 * i + 4],
                                                     "little") for i in range(4)]
                    l0, l1, cw = word(0), word(512), word(1024)
                    r0 = decode8(l0[0], l0[1], cw[0], base4) + decode8(l0[2], l0[3], cw[1], base4)
                    r8 = decode8(l1[0], l1[1], cw[2], base4) + decode8(l1[2], l1[3], cw[3], base4)
                    for e in lst:
                        if e >> 17 == s and ((e >> 12) & 31) == lane:
                            reg, mask = (e >> 8) & 15, (e & 127) << (8 + 16 * ((e >> 7) & 1))
                            if reg < 8:
                                r0[reg] ^= mask
                            else:
                                r8[reg - 8] ^= mask
                    row = 16 * rg + gq
                    for b in range(4):
                        frag = [r0[2 * b], r8[2 * b], r0[2 * b + 1], r8[2 * b + 1]]
                        for f, (dr, dk) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
                            kk = 64 * u + 16 * b + 2 * a + dk
                            for h in range(2):
                                got = (frag[f] >> (16 * h)) & 0xFFFF
                                bad += got != int(bits[row + dr, kk + h])
    return pk["escapes"], pk["max_esc"], bad


if __name__ == "__main__":
    fails = 0
    for n, clk, ku, seed in ((128, 4, 2, 1), (256, 2, 3, 2)):
        nesc, mx, bad = check(n, clk, ku, seed)
        print(f"n {n} clk {clk} ku {ku}: {nesc} escapes (max {mx} per warp), {bad} mismatched weights")
        fails += bad
    sys.exit(1 if fails else 0)
