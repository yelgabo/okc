#!/usr/bin/env python3
"""pdl8f's staged gate on stress inputs without routing near ties: writes muse_decode_layer/cand/g1b4_pdl8g.py.

The file is g1b4_pdl8f.py (tools/dl_pdl8f_patch.py, c4164566) with one change in prepare. In check 9ae1d02f0c2e4244 the
T = 16 fresh stage failed its ids-equal clause on merge-order noise (the PDL-off arm alone flipped 52 ids), because a
random stress input routes some tokens with the 8th and 9th expert scores almost tied. Each stress input (x0 and the
alternating x1, at every T) is now built so that every token's gap between its 8th and 9th router selection score
(sigmoid(logit / sqrt(32)) + bias, as the route kernels compute it) is at least PDL8G_MARGIN. Rows below the margin are
resampled, and before use the input must keep identical ids over PDL8G_STABLE PDL-off graph replays. The scores are
read in eager calls during construction only, by wrapping the Python launchers of the route kernels; no CUDA source,
launch attribute or timed path changes. Stages, bound (ids equal, floats within max(2e-2, off)), per-T abort-to-off and
fallbacks are pdl8f's; stage lines also print the ids-mismatch ratio on / off as a diagnostic.

    python3 tools/dl_pdl8g_patch.py
"""
import hashlib
import importlib.util
import py_compile
import sys

spec = importlib.util.spec_from_file_location("pdl8f", __file__.replace("dl_pdl8g_patch.py", "dl_pdl8f_patch.py"))
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
E, D, B = F.E, F.D, F.B

DST = B.A.ROOT / "muse_decode_layer/cand/g1b4_pdl8g.py"
PDL8F_PIN = "c4164566f0315cfc9059558b1531bf0b9e1ce844fd137d437df355b7ccb509cc"

HELPERS = r'''# Merge-order noise moves router selection scores by far less than 1e-3 (it flips BF16 activations by one ulp); a
# gap of 1e-3 between the 8th and 9th score keeps a token's ids fixed under that noise.
PDL8G_MARGIN = 1e-3
PDL8G_ROUNDS = 24
PDL8G_STABLE = 50
_PDL8G_ROUTES = ("_sl_tile_route", "_sl_tile_route_cg", "_graft_route", "_sl_graft_route", "_gle_route", "_route")


def _pdl8g_sel(name, grid, a, kw):
    """The route kernel's selection scores [T, 256] from its launch arguments, or None for an unknown layout."""
    if name == "_gle_route":
        T = int(a[3])
        part = a[2].reshape(-1)[:256 * T * 8].view(256, T, 8).float().sum(0)
        r1 = torch.rsqrt(part[:, 0] / 8192.0 + 1e-8)
        r2 = torch.rsqrt((part[:, 1] + 2.0 * r1 * part[:, 2] + r1 * r1 * part[:, 3]) / 8192.0 + 1e-5)
        lg = a[1].reshape(-1)[:2 * T * 256].view(2, T, 256).float()
        logit = (lg[0] + lg[1] * r1[:, None]) * r2[:, None]
        bias = a[4]
    elif "SPLIT" in kw:
        T = int(grid[0])
        if kw["SPLIT"]:
            logit = a[1].reshape(-1)[:T * 256].view(T, 256).float()
        else:
            al = a[2].reshape(-1)[:T * 4608].view(T, 4608).float()
            logit = al[:, 4096:4352] + al[:, 4352:4608]
        bias = a[3]
    else:
        return None
    return torch.sigmoid(logit * ROUTER_MULT) + bias.float().view(1, 256)


def _pdl8g_gaps(data, state):
    """Per-token gap between the 8th and 9th selection score of one eager PDL-off call."""
    g = globals()
    saved = {n: g[n] for n in _PDL8G_ROUTES if n in g}
    seen = []

    class _Rec:
        def __init__(self, name, k):
            self.name, self.k = name, k

        def __getitem__(self, grid):
            f = self.k[grid]

            def run(*a, **kw):
                out = f(*a, **kw)
                sel = _pdl8g_sel(self.name, grid, a, kw)
                seen.append(None if sel is None else sel.clone())
                return out
            return run

    try:
        for n, k in saved.items():
            g[n] = _Rec(n, k)
        kernel_fn(data, state)
        torch.cuda.synchronize()
    finally:
        g.update(saved)
    if len(seen) != 1 or seen[0] is None:
        return None
    top = seen[0].topk(9, dim=-1).values
    return top[:, 7] - top[:, 8]


def _pdl8g_unit_rows(n, cols, gen, dev, dtype):
    s = torch.randn((n, cols), generator=gen, device=dev)
    return (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype)


def _pdl8g_tiefree(st, T, data, x, gen, dev, tag):
    """x with every token's 8th-to-9th score gap at least PDL8G_MARGIN and ids stable over PDL-off replays."""
    off = dict(st, pdl8_on={})
    keep = data["attn_input"].clone()
    try:
        data["attn_input"].copy_(x)
        for r in range(PDL8G_ROUNDS):
            gaps = _pdl8g_gaps(data, off)
            if gaps is None:
                raise RuntimeError("router selection scores not observable on this path")
            low = (gaps < PDL8G_MARGIN).nonzero().flatten()
            if low.numel() == 0:
                break
            data["attn_input"][low] = _pdl8g_unit_rows(low.numel(), x.shape[1], gen, dev, x.dtype)
        else:
            raise RuntimeError(f"no input without near ties after {PDL8G_ROUNDS} rounds")
        g_off, o_off = _pdl8_graph(lambda: kernel_fn(data, off))
        g_off.replay()
        torch.cuda.synchronize()
        ref = [t.clone() for t in o_off]
        _, moved = _pdl8_drift(g_off, o_off, ref, PDL8G_STABLE)
        del g_off, o_off, ref
        if moved:
            raise RuntimeError(f"ids moved on {moved} entries over {PDL8G_STABLE} PDL-off replays")
        print(f"PDL8G_INPUT T={T} {tag}: min 8th-9th gap {float(gaps.min()):.2e} after {r} resample rounds; ids stable "
              f"over {PDL8G_STABLE} PDL-off replays", flush=True)
        return data["attn_input"].clone()
    finally:
        data["attn_input"].copy_(keep)


'''

T8_X_OLD = '''            x0 = data["attn_input"].clone()
            s = torch.randn(x0.shape, generator=gen, device=dev)
            x1 = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x0.dtype)
            datas[layer] = (data, ((x0, x1), torch.zeros((1, 1), dtype=torch.bool, device=dev)))
'''
T8_X_NEW = '''            x0 = _pdl8g_tiefree(st, 8, data, data["attn_input"].clone(), gen, dev, f"{layer} x0")
            data["attn_input"].copy_(x0)
            x1 = _pdl8g_tiefree(st, 8, data, _pdl8g_unit_rows(x0.shape[0], x0.shape[1], gen, dev, x0.dtype), gen, dev,
                                f"{layer} x1")
            datas[layer] = (data, ((x0, x1), torch.zeros((1, 1), dtype=torch.bool, device=dev)))
'''
GATE_X_OLD = '''    x0 = data["attn_input"].clone()
    s = torch.randn(x0.shape, generator=gen, device=dev)
    x1 = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x0.dtype)
    flag = torch.zeros((1, 1), dtype=torch.bool, device=dev)
'''
GATE_X_NEW = '''    x0 = _pdl8g_tiefree(st, T, data, data["attn_input"].clone(), gen, dev, f"{layer} x0")
    data["attn_input"].copy_(x0)
    x1 = _pdl8g_tiefree(st, T, data, _pdl8g_unit_rows(x0.shape[0], x0.shape[1], gen, dev, x0.dtype), gen, dev,
                        f"{layer} x1")
    flag = torch.zeros((1, 1), dtype=torch.bool, device=dev)
'''
T8_PRINT_OLD = '''f"bound max({PDL8F_BOUND:.0e}, off): "
              f"{'clean' if ok else 'DRIFT'} ({spent[-1]:.1f} s)", flush=True)'''
T8_PRINT_NEW = '''f"bound max({PDL8F_BOUND:.0e}, off): "
              f"{'clean' if ok else 'DRIFT'} ({spent[-1]:.1f} s); ids mismatch on/off {on_bad}/{off_bad}", flush=True)'''
GATE_PRINT_OLD = '''f"{'clean' if good else 'DRIFT, T off'}", flush=True)'''
GATE_PRINT_NEW = '''f"{'clean' if good else 'DRIFT, T off'}; ids mismatch on/off {on_bad}/{off_bad}", flush=True)'''

EDITS = [
    ("tools/dl_pdl8f_patch.py).", "tools/dl_pdl8f_patch.py), on stress inputs without routing near ties "
     "(tools/dl_pdl8g_patch.py).", 1),
    ("def _pdl8e_drift2(", HELPERS + "def _pdl8e_drift2(", 1),
    (T8_X_OLD, T8_X_NEW, 1),
    (GATE_X_OLD, GATE_X_NEW, 1),
    (T8_PRINT_OLD, T8_PRINT_NEW, 1),
    (GATE_PRINT_OLD, GATE_PRINT_NEW, 1),
]


def main():
    s, body = B.make_block()
    for mod in (D, E, F):
        for old, new, n in mod.EDITS:
            body = B.A.sub(body, old, new, n)
    if hashlib.sha256((s.rstrip("\n") + "\n" + body).encode()).hexdigest() != PDL8F_PIN:
        sys.exit("tools/dl_pdl8f_patch.py no longer reproduces g1b4_pdl8f.py c4164566")
    cuda_before = body[body.index("_PDL8_EDITS = "):body.index("_PDL8_GRAFT_GLUE_EDITS")]
    for old, new, n in EDITS:
        body = B.A.sub(body, old, new, n)
    if cuda_before not in body or body.count("launch_pdl=_PDL8)") != 2:
        sys.exit("pdl8g changed a CUDA edit or a T = 16 / 32 launch")
    DST.write_text(s.rstrip("\n") + "\n" + body)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
