"""CPU work model for a denser lossless BF16 weight code (muse_bf16_proj, m <= 32 streams).

Usage: uv run --with torch python tools/proj_codec_model.py [--seeds 5151,1,2] [--json out.json]

Weights follow reference.py's generate_weights formula (randn / sqrt(K), cast to BF16). The
judge's generator is torch's CUDA Philox, which this machine cannot run, so the CPU generator
gives the same distribution, not the same bits; the judge also redraws weights per run (its
pack lines report 2583 to 2656 qkv_gate escapes across receipts), so per-run variation is
already part of the judged signal. reference.py is not imported: the formula is restated here.

For each scheme it reports physical bits per weight (code + escape entries), escapes per
weight, per-CTA escape counts at the shipped sk geometries, and a bit-exact round trip of every
weight through the scheme's encoder and decoder.
"""
import argparse
import json
import math

import torch

LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
UNPOOL_SLICE = 1024
# Rows per CTA of the shipped sk picks (SK_GEOM: R = NWR * NT * 8) and of cgemv (2 rows).
CTA_ROWS = {
    "qkv_gate": {"cgemv": 2, "qkv8": 24, "qkv16": 24, "qkv32": 24},
    "o": {"cgemv": 2, "o8": 32, "o16": 32, "o32": 32},
    "pooler": {"cgemv": 2, "pool8": 16, "pool16": 32, "pool32": 32},
    "unpooler": {"cgemv_slice": 2, "slice8": 8, "unp16": 32, "unp32": 32},
}


def gen_weights(seed):
    g = torch.Generator().manual_seed(seed)
    return {name: (torch.randn((n, k), generator=g) / math.sqrt(k)).to(torch.bfloat16)
            for name, (n, k) in LAYERS.items()}


def bits_of(w):
    return w.view(torch.int16).to(torch.int32) & 0xFFFF


def entropy(counts):
    p = counts[counts > 0].double() / counts.sum()
    return float(-(p * p.log2()).sum())


def window(hist, width):
    """Start of the width-wide window of hist with the most mass."""
    return int(hist.unfold(0, width, 1).sum(1).argmax())


def scheme_cur12(bits):
    """Shipped code: raw low byte (exponent bit 0 + mantissa), sign + 3-bit offset of the top
    seven exponent bits in an 8-wide window. Escape = top bits outside the window."""
    e71 = (bits >> 8) & 0x7F
    base = window(torch.bincount(e71.flatten(), minlength=128), 8)
    off = (e71 - base).clamp(0, 7)
    esc = (e71 < base) | (e71 > base + 7)
    dec = (bits & 0x80FF) | ((base + off) << 8)
    return 12.0, esc, dec, {"base": base}


def scheme_p11(bits):
    """Shipped layout with a 2-bit offset: raw low byte, sign + 2-bit offset of the top seven
    exponent bits (4 pairs = 8 binades, pair-aligned)."""
    e71 = (bits >> 8) & 0x7F
    base = window(torch.bincount(e71.flatten(), minlength=128), 4)
    off = (e71 - base).clamp(0, 3)
    esc = (e71 < base) | (e71 > base + 3)
    dec = (bits & 0x80FF) | ((base + off) << 8)
    return 11.0, esc, dec, {"base": base}


def scheme_b11(bits):
    """Byte = sign + 7 mantissa bits, 3-bit offset of the full exponent (8 binades at any
    alignment)."""
    e = (bits >> 7) & 0xFF
    base = window(torch.bincount(e.flatten(), minlength=256), 8)
    off = (e - base).clamp(0, 7)
    esc = (e < base) | (e > base + 7)
    dec = (bits & 0x807F) | ((base + off) << 7)
    return 11.0, esc, dec, {"base": base}


def scheme_lut(bits, group, cbits):
    """Exponents of `group` consecutive-k weights coded jointly by cbits through a LUT of the
    2^cbits most frequent exponent tuples; sign + mantissa stay as one raw byte per weight.
    A tuple outside the LUT decodes as the LUT entry chosen per weight by clamping each exponent
    into the per-weight window [lo, hi] of the LUT's entries, and every weight whose exponent
    differs is an escape. group = 2 matches one mma B register (a bf16x2 of adjacent k)."""
    n, k = bits.shape
    e = ((bits >> 7) & 0xFF).view(n, k // group, group)
    emode_hist = torch.bincount(e.flatten(), minlength=256)
    lo = window(emode_hist, 16)
    ec = (e - lo).clamp(0, 15)
    idx = torch.zeros(ec.shape[:2], dtype=torch.int64)
    for j in range(group):
        idx = idx * 16 + ec[..., j].long()
    tup_hist = torch.bincount(idx.flatten(), minlength=16 ** group)
    # Tuples with an exponent outside the 16-binade frame are never LUT entries.
    outside = ((e < lo) | (e > lo + 15)).any(-1)
    tup_hist_in = tup_hist.clone()
    tup_hist_in -= torch.bincount(idx[outside].flatten(), minlength=16 ** group)
    lut = tup_hist_in.argsort(descending=True)[: 2 ** cbits]
    in_lut = torch.zeros(16 ** group, dtype=torch.bool)
    in_lut[lut] = True
    hit = in_lut[idx] & ~outside
    # A miss decodes as the LUT entry that matches the most of its weights' exponents (exponents
    # outside the 16-binade frame are clamped into it first); every mismatching weight is an
    # escape. The miss -> entry map is a second table built in prepare().
    digits = lambda t: torch.stack([(t // 16 ** (group - 1 - j)) % 16 for j in range(group)], -1)
    lut_d = digits(lut)
    seen = torch.unique(idx[~hit])
    best = torch.empty(16 ** group, dtype=torch.int64)
    for c0 in range(0, seen.numel(), 4096):
        sd = digits(seen[c0:c0 + 4096])
        match = (sd[:, None, :] == lut_d[None, :, :]).sum(-1)
        best[seen[c0:c0 + 4096]] = lut[match.argmax(1)]
    fb = torch.where(hit, idx, best[idx])
    dec_e = digits(fb).to(torch.int32) + lo
    esc = (dec_e != e).view(n, k)
    dec = (bits & 0x807F) | (dec_e.view(n, k) << 7)
    return 8.0 + cbits / group, esc, dec, {"lo": lo, "tuple_miss": float((~hit).double().mean())}


SCHEMES = {
    "cur12": scheme_cur12,
    "p11": scheme_p11,
    "b11": scheme_b11,
    "pair8": lambda b: scheme_lut(b, 2, 8),
    "pair7": lambda b: scheme_lut(b, 2, 7),
    "pair6": lambda b: scheme_lut(b, 2, 6),
    "pair5": lambda b: scheme_lut(b, 2, 5),
    "quad12": lambda b: scheme_lut(b, 4, 12),
    "quad11": lambda b: scheme_lut(b, 4, 11),
    "quad10": lambda b: scheme_lut(b, 4, 10),
    "quad13": lambda b: scheme_lut(b, 4, 13),
    "quad14": lambda b: scheme_lut(b, 4, 14),
}


def roundtrip_exact(bits, esc, dec):
    """Decoded bits with every escape restored from its stored original must equal the input."""
    restored = torch.where(esc, bits, dec)
    return bool(torch.equal(restored, bits))


def per_cta(esc, rows):
    n = esc.shape[0]
    c = esc[: (n // rows) * rows].view(n // rows, -1).sum(1)
    return int(c.max()), float(c.double().mean())


def analyze(seed):
    out = {}
    ws = gen_weights(seed)
    for layer, w in ws.items():
        bits = bits_of(w)
        exp_hist = torch.bincount(((bits >> 7) & 0xFF).flatten(), minlength=256)
        mant_hist = torch.bincount((bits & 0x7F).flatten(), minlength=128)
        h_e, h_m = entropy(exp_hist), entropy(mant_hist)
        lay = {"n": int(bits.numel()), "H_exp": h_e, "H_mant": h_m, "H_total": 1 + h_e + h_m, "schemes": {}}
        for name, fn in SCHEMES.items():
            code_bits, esc, dec, info = fn(bits)
            rate = float(esc.double().mean())
            s = {"code_bits": code_bits, "esc_rate": rate, "exact": roundtrip_exact(bits, esc, dec), **info,
                 "eff_bits_e32": code_bits + 32 * rate, "eff_bits_e64": code_bits + 64 * rate, "cta": {}}
            for geo, rows in CTA_ROWS[layer].items():
                sub = esc[:UNPOOL_SLICE] if geo in ("slice8", "cgemv_slice") else esc
                s["cta"][geo] = per_cta(sub, rows)
            lay["schemes"][name] = s
        out[layer] = lay
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="5151,1,2")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        res[seed] = analyze(seed)
        for layer, lay in res[seed].items():
            print(f"seed {seed} {layer}: H exp {lay['H_exp']:.3f} mant {lay['H_mant']:.3f} total {lay['H_total']:.3f}")
            for name, s in lay["schemes"].items():
                cta = " ".join(f"{g} {mx}/{mean:.1f}" for g, (mx, mean) in s["cta"].items())
                print(f"  {name:7s} code {s['code_bits']:5.2f} esc {s['esc_rate']:.3e} eff32 {s['eff_bits_e32']:.3f}"
                      f" eff64 {s['eff_bits_e64']:.3f} exact {s['exact']} | per-CTA max/mean {cta}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
