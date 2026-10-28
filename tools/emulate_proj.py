"""CPU emulation of sk_kernel (muse_bf16_proj/submission.py) on small problems.

The submission packs the weights. A numpy byte array stands in for each CTA's shared memory,
filled as the bulk copy and the 128B-swizzled TMA box fill it, and every read goes through the
kernel's own address arithmetic: ldmatrix.x4, decode8's prmt, the mma.m16n8k16 fragment
layouts, the partial-sum and peer buffers, the escape warps' table walk, the row map and the
fused router tail. Products are summed in float64, so a mismatch against x @ W.T is a packing
or indexing error, not rounding.

    uv run --with torch --with numpy python tools/emulate_proj.py
"""

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MS = {8: (1, 3, 8), 16: (9, 16), 32: (17, 32)}
NTINY, NHUGE = 6, 2
FUSED_PIECES = 2
SEED = 2026

LANE = np.arange(32)
G, T4 = LANE >> 2, LANE & 3
LROW = (LANE & 7) + ((LANE >> 3) & 1) * 8
LCHUNK = LANE >> 4
REG = np.arange(4)[:, None]
# mma.m16n8k16 fragments as the PTX ISA documents them, g = lane>>2, t = lane&3: A register i
# holds (row g + 8(i&1), k 2t + 8(i>>1) + {0,1}), B register i holds (k 2t + 8i + {0,1}, column g),
# C register e holds (row g + 8(e>>1), column 2t + (e&1)).
AROW, ACOL = G[None, :] + 8 * (REG & 1), 2 * T4[None, :] + 8 * (REG >> 1)
BROW, BCOL = 2 * T4[:, None] + 8 * np.arange(2)[None, :], np.broadcast_to(G[:, None], (32, 2))
EL = np.arange(4)[None, :]
CROW, CCOL = G[:, None] + 8 * (EL >> 1), 2 * T4[:, None] + (EL & 1)
# ldmatrix.x4: lanes 8i..8i+7 supply the row addresses of matrix i; register i of lane l holds
# elements 2(l&3), 2(l&3)+1 of row l>>2 of matrix i.
LDSM_ROW = 8 * REG + G[None, :]


def load_submission():
    spec = importlib.util.spec_from_file_location("submission", ROOT / "muse_bf16_proj" / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Cfg:
    """SkCfg's constants for one SK_CONFIGS row."""

    def __init__(self, row):
        (self.name, self.MT, self.TOKR, self.NT, self.NWR, self.WSK, self.UPW, self.NSTAGE, self.COMP,
         self.HL, self.NACC, self.CL, self.out) = row[:13]
        self.FUSE = bool(row[13]) if len(row) > 13 else False
        self.NW = self.NWR * self.WSK
        self.NC = self.NW * 32
        self.R = self.NWR * self.NT * 8
        self.UPS = self.WSK * self.UPW
        self.TU = 768 if self.COMP else (2048 if self.HL else 1024)
        self.WBY = self.NW * self.UPW * self.NT * self.TU
        self.XBY = self.UPS * self.TOKR * 128
        self.RUPW = self.UPS // self.NW if self.FUSE else 1
        self.RWBY = self.UPS * 2048 if self.FUSE else 0
        self.SST = (self.XBY + max(self.WBY, self.RWBY) + 1023) & ~1023
        self.TOKS = 16 * self.MT
        self.RP = self.R + ((40 - (self.R % 32)) % 32)
        self.CP = self.R + 1
        self.NESC = (4 if self.NSTAGE >= 4 else self.NSTAGE) if self.COMP else 0
        self.CSZ = (self.TOKS * self.CP + 3) & ~3
        self.KST = self.UPS * 64
        self.HM = 2 if self.HL else 1
        assert self.WSK * self.TOKS * self.RP * 4 <= self.NSTAGE * self.SST, "partials must fit in the ring"
        assert self.TOKR == 16 * self.MT or (self.TOKR == 8 and self.MT == 1)
        assert not (self.COMP and self.CL > 1)
        assert not self.FUSE or (self.COMP and self.CL == 1 and self.NW * self.RUPW == self.UPS
                                 and self.NW * self.TOKS * 8 * 4 <= self.NSTAGE * self.SST)

    def split(self, crank, tst):
        gs0 = (crank * tst) // self.CL
        return gs0, ((crank + 1) * tst) // self.CL - gs0


class Smem:
    def __init__(self, nbytes):
        self.mem = np.zeros(nbytes, np.uint8)

    def u32(self, addr):
        b = self.mem[np.asarray(addr)[..., None] + np.arange(4)].astype(np.uint32)
        return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)

    def u16(self, addr):
        b = self.mem[np.asarray(addr)[..., None] + np.arange(2)].astype(np.uint32)
        return b[..., 0] | (b[..., 1] << 8)


def prmt(a, b, sel):
    pool = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    d = np.zeros_like(a)
    for i in range(4):
        d |= pool[(sel >> (4 * i)) & 7] << (8 * i)
    return d


def decode8(lw0, lw1, codes, base4):
    ha = ((codes & 0x07070707) + base4) | ((codes << 4) & 0x80808080)
    hb = (((codes >> 4) & 0x07070707) + base4) | (codes & 0x80808080)
    return [prmt(lw0, ha, 0x5140), prmt(lw0, ha, 0x7362), prmt(lw1, hb, 0x5140), prmt(lw1, hb, 0x7362)]


def bf16_f64(h):
    return (h.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def split_regs(regs):
    return bf16_f64(regs & 0xFFFF), bf16_f64(regs >> 16)


def frag_a(a):
    """A fragments [..., 4, 32] -> [..., 16, 16]."""
    A = np.zeros(a.shape[:-2] + (16, 16))
    lo, hi = split_regs(a)
    A[..., AROW, ACOL] = lo
    A[..., AROW, ACOL + 1] = hi
    return A


def frag_b(b):
    """B fragments [..., 32, 2] -> [..., 16, 8]."""
    B = np.zeros(b.shape[:-2] + (16, 8))
    lo, hi = split_regs(b)
    B[..., BROW, BCOL] = lo
    B[..., BROW + 1, BCOL] = hi
    return B


def frag_c(C):
    """[..., 16, 8] -> C fragments [..., 32, 4]."""
    return C[..., CROW, CCOL]


def b_slices(b):
    """[..., 32, 8] registers -> [..., 4 k-slices, 32, 2]: slice j is registers 2j, 2j+1."""
    return np.moveaxis(b.reshape(b.shape[:-1] + (4, 2)), -2, -3)


def load_a(smem, xt, ui, M, MT, zrow):
    """The kernel's load_a for unit bases xt[...] and unit indices ui[...]: [..., 4 j, MT, 4, 32]."""
    a = np.zeros(xt.shape + (4, MT, 4, 32), np.uint32)
    for mt in range(MT):
        trow = mt * 16 + LROW
        rbase = xt[..., None] + trow * 128
        sw = (ui[..., None] * M + trow) & 7
        for j in range(4):
            addr = np.where(trow < M, rbase + (((2 * j + LCHUNK) ^ sw) << 4), zrow)
            a[..., j, mt, :, :] = smem.u32(addr[..., LDSM_ROW] + 4 * T4[None, :])
    return a


def fill_x_box(smem, s0, xbytes, M, unit0, UPS):
    lanes = np.arange(M)
    for u in range(UPS):
        r = u * M + lanes
        col0 = (unit0 + u) * 128
        for ch in range(8):
            dst = s0 + r * 128 + ((ch ^ (r & 7)) << 4)
            smem.mem[dst[:, None] + np.arange(16)] = xbytes[:, col0 + ch * 16:col0 + ch * 16 + 16]


def run_cta(c, b, nrb, stream, rstream, xbytes, M, K, tab, base4):
    """One CTA (blockIdx.x = b) through all its stages. Returns its summed accumulators
    [NW, MT, NT, 32, 4], corr [NESC, CSZ] and the fused router accumulators [2, NW, MT, 32, 4]."""
    rb, crank = (b // c.CL, b % c.CL) if c.CL > 1 else (b, 0)
    tst = K // c.KST
    gs0, nst = c.split(crank, tst)
    npiece = nrb // 32 if c.FUSE else 1
    spp = tst // npiece if c.FUSE else 0
    rbk, rq = (b % 32, b // 32) if c.FUSE else (0, 0)
    nst_tot = nst + spp
    wstride = nrb * c.WBY
    XU = M * 128
    smem = Smem(c.NSTAGE * c.SST + 128)
    zrow = c.NSTAGE * c.SST
    acc = np.zeros((c.NACC, c.NW, c.MT, c.NT, 32, 4))
    racc = np.zeros((2, c.NW, c.MT, 32, 4))
    corr = np.zeros((c.NESC, c.CSZ), np.float32)
    pe = [1] * c.NESC
    cnt = int(tab[0]) if c.COMP else 0
    tabf = tab.view(np.float32) if c.COMP else None
    lanes = np.arange(M)
    WI, UI = np.arange(c.NW)[:, None], np.arange(c.UPW)[None, :]
    RUI = np.arange(c.RUPW)[None, :]
    NTI = np.arange(c.NT)
    for st in range(nst_tot):
        gst = gs0 + st
        s0 = (st % c.NSTAGE) * c.SST
        if st >= nst:
            rs = rq * spp + (st - nst)
            src = (rs * 32 + rbk) * c.RWBY
            smem.mem[s0 + c.XBY:s0 + c.XBY + c.RWBY] = rstream[src:src + c.RWBY]
            fill_x_box(smem, s0, xbytes, M, rs * c.UPS, c.UPS)
            ui = WI * c.RUPW + RUI
            wb = s0 + c.XBY + ui * 2048
            bh = smem.u32(wb[..., None, None] + LANE[:, None] * 32 + 4 * np.arange(8))
            bl = smem.u32(wb[..., None, None] + 1024 + LANE[:, None] * 32 + 4 * np.arange(8))
            A = frag_a(load_a(smem, s0 + ui * XU, ui, M, c.MT, zrow))
            for h, breg in enumerate((bh, bl)):
                C = np.einsum("wujmik,wujkc->wujmic", A, frag_b(b_slices(breg)))
                racc[h] += frag_c(C).sum(axis=(1, 2))
            continue
        src = gst * wstride + rb * c.WBY
        smem.mem[s0 + c.XBY:s0 + c.XBY + c.WBY] = stream[src:src + c.WBY]
        fill_x_box(smem, s0, xbytes, M, gst * c.UPS, c.UPS)
        ui = (WI % c.WSK) * c.UPW + UI
        tb = (s0 + c.XBY + (WI * c.UPW + UI) * c.NT * c.TU)[..., None] + NTI * c.TU
        if c.COMP:
            lw = smem.u32(tb[..., None, None] + LANE[:, None] * 16 + 4 * np.arange(4))
            cw = smem.u32(tb[..., None, None] + 512 + LANE[:, None] * 8 + 4 * np.arange(2))
            b_ = np.stack(decode8(lw[..., 0], lw[..., 1], cw[..., 0], base4)
                          + decode8(lw[..., 2], lw[..., 3], cw[..., 1], base4), axis=-1)
        else:
            b_ = smem.u32(tb[..., None, None] + LANE[:, None] * 32 + 4 * np.arange(8))
            if c.HL:
                bl = smem.u32(tb[..., None, None] + 1024 + LANE[:, None] * 32 + 4 * np.arange(8))
        A = frag_a(load_a(smem, s0 + ui * XU, ui, M, c.MT, zrow))
        C = frag_c(np.einsum("wujmik,wunjkc->wujmnic", A, frag_b(b_slices(b_))))
        if c.HL:
            CL_ = frag_c(np.einsum("wujmik,wunjkc->wujmnic", A, frag_b(b_slices(bl))))
        for uu in range(c.UPW):
            for j in range(4):
                acc[((uu * 4 + j) * c.HM) % c.NACC] += C[:, uu, j]
                if c.HL:
                    acc[((uu * 4 + j) * c.HM + 1) % c.NACC] += CL_[:, uu, j]
        if not c.COMP:
            continue
        kbeg, kend = gst * c.KST, (gst + 1) * c.KST
        for ew in range(c.NESC):
            if st % c.NESC != ew:
                while pe[ew] <= cnt and (int(tab[2 * pe[ew]]) & 0xFFFF) < kend:
                    pe[ew] += 1
                continue
            while pe[ew] <= cnt:
                e0 = int(tab[2 * pe[ew]])
                kk = e0 & 0xFFFF
                if kk >= kend:
                    break
                kl = kk - kbeg
                u, kc = kl >> 6, kl & 63
                xa = s0 + u * XU + lanes * 128 + (((kc >> 3) ^ ((u * M + lanes) & 7)) << 4) + (kc & 7) * 2
                idx = lanes * c.CP + (e0 >> 16)
                # nvcc contracts the update into one FFMA, a single rounding of the exact sum.
                corr[ew, idx] = (corr[ew, idx].astype(np.float64)
                                 + np.float64(tabf[2 * pe[ew] + 1]) * bf16_f64(smem.u16(xa))).astype(np.float32)
                pe[ew] += 1
    assert all(p == cnt + 1 for p in pe), "escape warps did not consume the whole table"
    return acc.sum(axis=0), corr, racc


def epilogue(c, rb, M, accs, corr, tab, pmax, out):
    """The partial-sum stores of every rank, then rank 0's final sum and output indexing."""
    nbuf = c.WSK * c.TOKS * c.RP
    red = np.zeros(nbuf)
    peer = np.zeros((c.CL - 1) * nbuf)
    written = np.zeros(c.CL * nbuf, bool)
    for crank in range(c.CL):
        dst = red if crank == 0 else peer[(crank - 1) * nbuf:crank * nbuf]
        wmark = written[crank * nbuf:(crank + 1) * nbuf]
        for warp in range(c.NW):
            wr, wk = warp // c.WSK, warp % c.WSK
            for mt in range(c.MT):
                for nt in range(c.NT):
                    row = (wr * c.NT + nt) * 8 + 2 * T4
                    tok = mt * 16 + G
                    r0 = (wk * c.TOKS + tok) * c.RP + row
                    for e, off in enumerate((0, 1, 8 * c.RP, 8 * c.RP + 1)):
                        assert not wmark[r0 + off].any(), "partial-sum slot written twice"
                        wmark[r0 + off] = True
                        dst[r0 + off] = accs[crank][warp, mt, nt, :, e]
    for idx in range(c.R // 4 * M):
        tt, row = idx // (c.R // 4), (idx % (c.R // 4)) * 4
        v = np.zeros(4)
        for e in range(c.NESC):
            v += corr[e, tt * c.CP + row:tt * c.CP + row + 4]
        for w in range(c.WSK):
            r0 = (w * c.TOKS + tt) * c.RP + row
            assert written[r0:r0 + 4].all(), "final sum reads an unwritten partial"
            v += red[r0:r0 + 4]
            for cc in range(c.CL - 1):
                q0 = ((cc * c.WSK + w) * c.TOKS + tt) * c.RP + row
                assert written[(cc + 1) * nbuf + q0:(cc + 1) * nbuf + q0 + 4].all(), "unwritten peer partial"
                v += peer[q0:q0 + 4]
        orow = int(tab[2 * pmax + row]) if c.COMP else rb * c.R + row
        out[tt, orow:orow + 4] = v


def fused_epilogue(c, b, nrb, M, racc, rpart, rcount, rout):
    """The router tail of a fused CTA: stride-8 partials in the ring, one piece of rpart, and
    the last arrival's sum over the pieces."""
    npiece, rbk, rq = nrb // 32, b % 32, b // 32
    red = np.zeros(c.NW * c.TOKS * 8)
    written = np.zeros(red.shape, bool)
    for warp in range(c.NW):
        for mt in range(c.MT):
            r0 = (warp * c.TOKS + mt * 16 + G) * 8 + 2 * T4
            for e, off in enumerate((0, 1, 64, 65)):
                assert not written[r0 + off].any(), "router partial written twice"
                written[r0 + off] = True
                red[r0 + off] = racc[0, warp, mt, :, e] + racc[1, warp, mt, :, e]
    for idx in range(2 * M):
        tt, row = idx >> 1, (idx & 1) * 4
        v = np.zeros(4)
        for w in range(c.NW):
            r0 = (w * c.TOKS + tt) * 8 + row
            assert written[r0:r0 + 4].all(), "router final sum reads an unwritten partial"
            v += red[r0:r0 + 4]
        rpart[(rq * M + tt) * 256 + rbk * 8 + row:(rq * M + tt) * 256 + rbk * 8 + row + 4] = v
    old = rcount[rbk]
    rcount[rbk] += 1
    if (old + 1) % npiece == 0:
        for idx in range(2 * M):
            tt, row = idx >> 1, (idx & 1) * 4
            v = np.zeros(4)
            for pc in range(npiece):
                v += rpart[(pc * M + tt) * 256 + rbk * 8 + row:(pc * M + tt) * 256 + rbk * 8 + row + 4]
            rout[tt, rbk * 8 + row:rbk * 8 + row + 4] = v


def emulate(c, stream, rstream, x, nrows, K, tabs, pmax, base4):
    M = x.shape[0]
    nrb = nrows // c.R
    xbytes = x.view(torch.int16).numpy().view(np.uint8).reshape(M, 2 * K)
    out = np.full((M, nrows), np.nan)
    rout = np.full((M, 256), np.nan) if c.FUSE else None
    rpart = np.full((nrb // 32) * M * 256, np.nan) if c.FUSE else None
    rcount = np.zeros(32, int)
    for rb in range(nrb):
        tab = tabs[rb] if c.COMP else None
        ranks = [run_cta(c, rb * c.CL + crank, nrb, stream, rstream, xbytes, M, K, tab, base4)
                 for crank in range(c.CL)]
        epilogue(c, rb, M, [a for a, _, _ in ranks], ranks[0][1], tab, pmax, out)
        if c.FUSE:
            fused_epilogue(c, rb, nrb, M, ranks[0][2], rpart, rcount, rout)
    assert not np.isnan(out).any(), "an output element was never written"
    if c.FUSE:
        assert not np.isnan(rout).any() and not np.isnan(rpart).any(), "a router output was never written"
        assert (rcount == nrb // 32).all()
    return out, rout


def check_stream_coverage(mod, c, geom, nrows, K):
    """Every tile the kernel's bulk copies and reads land on is the one stage_stream meant for
    that (CTA, warp, unit), and each CTA sees each of its R rows exactly once over all K."""
    nunits = K // 64
    ntiles, nrb, tst = nrows // 8, nrows // c.R, nunits // c.UPS
    ids = torch.stack(torch.meshgrid(torch.arange(ntiles), torch.arange(nunits), indexing="ij"), -1)
    s = mod.stage_stream(ids, geom, nrows).numpy().reshape(-1, 2)
    per_stage = c.NW * c.UPW * c.NT
    assert s.shape[0] == tst * nrb * per_stage
    for rb in range(nrb):
        seen = set()
        for crank in range(c.CL):
            gs0, nst = c.split(crank, tst)
            for st in range(nst):
                gst = gs0 + st
                for warp in range(c.NW):
                    for uu in range(c.UPW):
                        for nt in range(c.NT):
                            tile, unit = s[(gst * nrb + rb) * per_stage + (warp * c.UPW + uu) * c.NT + nt]
                            assert tile == rb * (c.R // 8) + (warp // c.WSK) * c.NT + nt, "stream tile order"
                            assert unit == gst * c.UPS + (warp % c.WSK) * c.UPW + uu, "stream unit order"
                            assert (tile, unit) not in seen, "tile read twice"
                            seen.add((tile, unit))
        assert seen == {(rb * (c.R // 8) + i, u) for i in range(c.R // 8) for u in range(nunits)}, "coverage"


def check_fused_coverage(mod, c, rgeom, nrb, K):
    """The fused router stream: CTA b reads piece b // 32 of row block b % 32, warp w units
    w * RUPW + uu; the pieces of a row block together cover all K once."""
    nunits = K // 64
    tst, npiece = nunits // c.UPS, nrb // 32
    spp = tst // npiece
    ids = torch.stack(torch.meshgrid(torch.arange(32), torch.arange(nunits), indexing="ij"), -1)
    s = mod.stage_stream(ids, rgeom, 256).numpy().reshape(-1, 2)
    assert s.shape[0] == tst * 32 * c.UPS
    for rbk in range(32):
        seen = set()
        for rq in range(npiece):
            for st in range(spp):
                rs = rq * spp + st
                for warp in range(c.NW):
                    for uu in range(c.RUPW):
                        ui = warp * c.RUPW + uu
                        tile, unit = s[(rs * 32 + rbk) * c.UPS + ui]
                        assert tile == rbk and unit == rs * c.UPS + ui, "fused router stream order"
                        assert (tile, unit) not in seen
                        seen.add((tile, unit))
        assert seen == {(rbk, u) for u in range(nunits)}, "fused router coverage"


def check_bookkeeping(c, M, pmax):
    box_bytes = 64 * M * c.UPS * 2
    assert box_bytes + c.WBY == c.UPS * M * 128 + c.WBY, "stage_tx"
    if c.FUSE:
        assert box_bytes + c.RWBY == c.UPS * M * 128 + c.RWBY, "fused stage_tx"
    if c.COMP:
        assert ((2 * pmax + c.R) * 4) % 16 == 0, "table bytes are not a bulk-copy multiple"
        assert pmax % 2 == 0


def gaussian_bf16(n, k, gen):
    return (torch.randn(n, k, generator=gen) / math.sqrt(k)).to(torch.bfloat16)


def force_escapes(w, rows, gen):
    """Tiny and huge weights inside the first `rows` rows so the escape path runs; returns the
    (row, k) of the tiny ones."""
    k = w.shape[1]
    tiny = [(int(torch.randint(rows, (1,), generator=gen)), int(torch.randint(k, (1,), generator=gen)))
            for _ in range(NTINY)]
    huge = [(int(torch.randint(rows, (1,), generator=gen)), int(torch.randint(k, (1,), generator=gen)))
            for _ in range(NHUGE)]
    for r, col in tiny:
        w[r, col] = w[r, col] * 2.0 ** -30
    for r, col in huge:
        w[r, col] = w[r, col] * 2.0 ** 20
    return tiny


def make_x(M, K, gen, esc_col):
    """Gaussian tokens; the last row probes the column of a forced escape and, when M > 1, row 0
    probes a random column."""
    x = torch.randn(M, K, generator=gen).to(torch.bfloat16)
    probes = {M - 1: esc_col}
    if M > 1:
        probes[0] = int(torch.randint(K, (1,), generator=gen))
    for row, col in probes.items():
        x[row] = 0
        x[row, col] = 8.0
    return x, probes


def to_bf16(a):
    return torch.as_tensor(np.asarray(a)).float().to(torch.bfloat16).double()


def check_bf16(out, ref, probes, escape, wdec, x):
    """bf16-rounded: relative L2, the probe rows exact outside escape columns. Unrounded: every
    element within 2^-6 relative plus the kernel's own error budget, 2^-22 of the escapes'
    clamped-weight products (each delta is an fp32 rounding of w - w_decoded)."""
    emu, want = to_bf16(out), to_bf16(ref)
    err = float((emu - want).norm() / want.norm())
    ndiff = int((emu != want).sum())
    perr, pdiff = 0.0, 0
    for row, col in probes.items():
        perr = max(perr, float((emu[row] - want[row]).norm() / want[row].norm()))
        keep = torch.ones(out.shape[1], dtype=torch.bool)
        if escape is not None:
            keep &= ~escape[:out.shape[1], col]
        pdiff += int((emu[row][keep] != want[row][keep]).sum())
    emu64 = torch.as_tensor(out)
    tol = 2.0 ** -6 * ref.abs() + 2.0 ** -24 * ref.abs().amax(dim=1, keepdim=True)
    if escape is not None:
        nrows = out.shape[1]
        tol += 2.0 ** -22 * (x.double().abs() @ (escape[:nrows].double() * wdec[:nrows].abs()).T)
    nbad = int(((emu64 - ref).abs() > tol).sum())
    ok = err < 1e-5 and perr < 1e-6 and pdiff == 0 and nbad == 0
    return ok, f"rel {err:.1e} probe {perr:.1e} diff {ndiff} probe_diff {pdiff} out_of_tol {nbad}"


def check_f32(out, ref_hl, ref_w):
    emu = torch.as_tensor(out).float().double()
    e1 = float((emu - ref_hl).norm() / ref_hl.norm())
    e2 = float((emu - ref_w).norm() / ref_w.norm())
    return e1 < 1e-6 and e2 < 1e-4, f"rel(hi+lo) {e1:.1e} rel(fp32) {e2:.1e}"


def run_group(mod, group, gen, balance=False, sliced=False, label=""):
    """One weight matrix shared by the configs of `group` (same R), emulated at every M."""
    c0 = group[0]
    kst = math.lcm(*[c.KST for c in group])
    npiece = FUSED_PIECES if c0.FUSE else 1
    tst = 7 if c0.CL > 1 else -(-(max(c.NSTAGE for c in group) + 2) // npiece) * npiece
    K = tst * kst
    nrows = (32 * npiece if c0.FUSE else 2 if c0.CL > 1 else 3) * c0.R
    n_full = nrows + c0.R if sliced else nrows
    geoms = {c.name: mod._sk_geom(c.name) for c in group}
    slices = {c.name: nrows for c in group}
    tabs, pmax, base4, escape, wdec, note = None, 0, 0, None, None, ""
    rstreams, hi, lo, W32 = {}, None, None, None
    if c0.out == "f32":
        W = torch.randn(n_full, K, generator=gen) / math.sqrt(K)
        hi, lo, streams = mod.pack_router(W, geoms)
        tiny = [(0, int(torch.randint(K, (1,), generator=gen)))]
    else:
        W = gaussian_bf16(n_full, K, gen)
        tiny = force_escapes(W, c0.R if balance else nrows, gen)
        if c0.COMP:
            packed = mod.pack_layer(W, geoms, slices, balance=balance)
            assert packed is not None, "pack_layer rejected the test matrix"
            streams, base4 = packed["streams"], packed["base4"]
            (tabt, pmax), = packed["tabs"].values()
            tabs = tabt.numpy()
            _, escape, decoded = mod.code_bits(mod.bf16_bits(W), packed["base"])
            wdec = (decoded << 16).view(torch.float32).double()
            counts = tabs[:, 0]
            note = f"esc {packed['escapes']} inexact {packed['inexact']} per-cta {counts.min()}..{counts.max()}"
        else:
            streams = mod.pack_raw_layer(W, geoms, slices)
    if c0.FUSE:
        # The same packing as _build_fused: the router as a hi + lo pair on the fused geometry.
        W32 = torch.randn(256, K, generator=gen) / math.sqrt(K)
        hi = W32.to(torch.bfloat16)
        lo = (W32 - hi.float()).to(torch.bfloat16)
        units = mod.hilo_units(mod.bf16_bits(hi), mod.bf16_bits(lo))
        for c in group:
            rgeom = mod._fused_router_geom(c.name)
            rstreams[c.name] = mod.stage_stream(units, rgeom, 256)
            check_fused_coverage(mod, c, rgeom, nrows // c.R, K)
    ok = True
    for c in group:
        check_stream_coverage(mod, c, geoms[c.name], nrows, K)
        stream = streams[c.name].numpy()
        assert stream.dtype == np.uint8 and stream.shape[0] == tst * (nrows // c.R) * c.WBY
        rstream = rstreams[c.name].numpy() if c.FUSE else None
        for M in MS[c.TOKR]:
            check_bookkeeping(c, M, pmax)
            x, probes = make_x(M, K, gen, tiny[0][1])
            t0 = time.time()
            out, rout = emulate(c, stream, rstream, x, nrows, K, tabs, pmax, base4)
            if c.out == "f32":
                good, msg = check_f32(out, x.double() @ (hi.double() + lo.double()).T, x.double() @ W.double().T)
            else:
                good, msg = check_bf16(out, x.double() @ W[:nrows].double().T, probes, escape, wdec, x)
            if c.FUSE:
                rgood, rmsg = check_f32(rout, x.double() @ (hi.double() + lo.double()).T, x.double() @ W32.double().T)
                good, msg = good and rgood, msg + " router " + rmsg
            ok &= good
            print(f"{c.name + label:14s} M={M:2d} N={nrows:4d} K={K:5d} stages={tst} {msg} {note} "
                  f"{time.time() - t0:5.1f}s {'ok' if good else 'FAIL'}", flush=True)
    return ok


def main():
    mod = load_submission()
    gen = torch.Generator().manual_seed(SEED)
    cfgs = {row[0]: Cfg(row) for row in mod.SK_CONFIGS}
    ok = True
    for c in cfgs.values():
        ok &= run_group(mod, [c], gen, sliced=c.name.startswith("slice"))
    ok &= run_group(mod, [cfgs["qkv8"], cfgs["qkv16"], cfgs["qkv32"]], gen, balance=True, label="/bal")
    print("ALL CHECKS PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
