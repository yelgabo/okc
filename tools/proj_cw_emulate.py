"""CPU check of the cw stream and decoder (muse_bf16_proj/submission.py, _CW_SRC): pack_cw on
random weights with extra escapes, then, for sampled (k tile, row block) blocks, the shared-
memory tile the 128 decoder threads write (decode8 on the stream words, the swizzled 16-byte
stores, the escape patches) against the tile TMA writes from the raw weights with the 128-byte
swizzle. Also checks the bitwise round trip of _unpack_cw on the whole matrix.

    uv run --with torch --with numpy python tools/proj_cw_emulate.py
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sub", ROOT / "muse_bf16_proj/experiments/cw_p3_submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)


def decode8(lw0, lw1, codes, base4):
    m = 0xFFFFFFFF
    ha = (((codes & 0x07070707) + base4) | ((codes << 4) & 0x80808080)) & m
    hb = ((((codes >> 4) & 0x07070707) + base4) | (codes & 0x80808080)) & m

    def prmt(a, b, sel):
        src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
        return sum(src[(sel >> (4 * i)) & 7] << (8 * i) for i in range(4))
    return [prmt(lw0, ha, 0x5140), prmt(lw0, ha, 0x7362), prmt(lw1, hb, 0x5140), prmt(lw1, hb, 0x7362)]


def words(b):
    return [int.from_bytes(bytes(b[4 * i:4 * i + 4]), "little") for i in range(len(b) // 4)]


def emulate_block(blockbytes, eby, base4):
    """The 16 KB decoded tile pair as bytes, as the kernel's decoder threads write it."""
    tile = bytearray(2 * 8192)
    low = blockbytes[:8192]
    cod = blockbytes[8192:12288]
    for r in range(128):
        rl = r & 63
        lw = []
        for j in range(4):
            lw += words(low[(j * 128 + r) * 16:(j * 128 + r) * 16 + 16])
        cw = []
        for h in range(2):
            cw += words(cod[(h * 128 + r) * 16:(h * 128 + r) * 16 + 16])
        trow = (r >> 6) * 8192 + rl * 128
        for c in range(8):
            o = decode8(lw[4 * (c >> 1) + 2 * (c & 1)], lw[4 * (c >> 1) + 2 * (c & 1) + 1], cw[c], base4)
            a = trow + ((c ^ (rl & 7)) << 4)
            tile[a:a + 16] = b"".join(v.to_bytes(4, "little") for v in o)
    esc = words(blockbytes[12288:12288 + eby])
    for e in esc[1:1 + esc[0]]:
        er, ek = e & 127, (e >> 7) & 63
        a = (er >> 6) * 8192 + (er & 63) * 128 + (((ek >> 3) ^ (er & 7)) << 4) + (ek & 7) * 2
        tile[a:a + 2] = (e >> 16).to_bytes(2, "little")
    return bytes(tile)


def tma_tile(bits, rb, kt):
    """128 rows x 64 k of bf16 bits as two 64-row TMA boxes with the 128-byte swizzle."""
    tile = bytearray(2 * 8192)
    for r in range(128):
        for k in range(64):
            v = int(bits[rb * 128 + r, kt * 64 + k])
            rl = r & 63
            a = (r >> 6) * 8192 + rl * 128 + (((k >> 3) ^ (rl & 7)) << 4) + (k & 7) * 2
            tile[a:a + 2] = v.to_bytes(2, "little")
    return bytes(tile)


def main():
    gen = torch.Generator().manual_seed(7)
    n, k = 512, 1024
    w = torch.randn((n, k), generator=gen) * 0.01
    # Out-of-window weights in both directions (tiny and huge), more than the judge's matrix has.
    tiny = torch.rand((n, k), generator=gen) < 2e-3
    big = torch.rand((n, k), generator=gen) < 5e-4
    w = torch.where(tiny, w * 1e-9, w)
    w = torch.where(big, w * 1e6, w)
    w = w.to(torch.bfloat16)
    packed, why = sub.pack_cw(w)
    assert packed is not None, why
    bits = sub.bf16_bits(w)
    print(f"pack_cw: base {packed['base']}, {packed['escapes']} escapes, max {packed['cmax']} per block, "
          f"eby {packed['eby']}")
    assert torch.equal(sub._unpack_cw(packed["stream"], n, k, packed["eby"], packed["base"]), bits)
    stream = packed["stream"].numpy().tobytes()
    bby = 12288 + packed["eby"]
    nrb, nkt = n // 128, k // 64
    checked = esc_blocks = 0
    for kt in range(nkt):
        for rb in range(nrb):
            blk = stream[(kt * nrb + rb) * bby:(kt * nrb + rb + 1) * bby]
            cnt = int.from_bytes(blk[12288:12292], "little")
            if cnt == 0 and (kt + rb) % 5:
                continue
            got = emulate_block(blk, packed["eby"], packed["base4"])
            assert got == tma_tile(bits, rb, kt), f"block kt={kt} rb={rb} differs"
            checked += 1
            esc_blocks += cnt > 0
    print(f"decoder emulation equals the TMA tile on {checked} blocks ({esc_blocks} with escapes)")
    # A block without escapes: in-window weights only (no zeros, no outliers).
    w0 = (torch.rand((128, 64), generator=gen) * 0.02 + 0.001).to(torch.bfloat16)
    p0, _ = sub.pack_cw(w0)
    s0 = p0["stream"].numpy().tobytes()
    assert int.from_bytes(s0[12288:12292], "little") == 0
    assert emulate_block(s0, p0["eby"], p0["base4"]) == tma_tile(sub.bf16_bits(w0), 0, 0)
    print("decoder emulation equals the TMA tile on a block without escapes")
    # A block with more escapes than the slot holds must be refused, not truncated.
    w2 = w.clone()
    w2[:64, :64] = torch.full((64, 64), 1e30, dtype=torch.bfloat16)
    p2, why2 = sub.pack_cw(w2)
    assert p2 is None, "overfull block accepted"
    print(f"overfull block refused: {why2}")


if __name__ == "__main__":
    sys.exit(main())
