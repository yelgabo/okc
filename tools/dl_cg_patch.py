"""Port the BF16 board's m = 1 GEMVs (C2, public 9eba96894ee1402e) into the public decode layer.

Usage: python3 tools/dl_cg_patch.py RELEASE C2 OUT
  RELEASE  muse_decode_layer/cand/g1b4_stage_release.py (ba9e327c, public 607eb52bdaf24160)
  C2       git show 3044ee7:muse_bf16_proj/submission.py (b2c4c096, public 9eba96894ee1402e)

The release bytes are kept; one block is appended. The block packs qkv_gate, o and the unpooler's
1024-row slice into C2's planar 12-bit row stream, compiles C2's GEMV source verbatim, and offers
cgemv to the layer's own bf16 tuner as one more m = 1 candidate. The tuner's cold-L2 graph replays
(sk2 and cgemv interleaved) and the layer's chain tuner choose, so a slower cgemv is never picked.
"""
import ast
import hashlib
import re
import sys

RELEASE_SHA = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
C2_SHA = 'b2c4c096410e2d282d5e1eb3f2afa083f61e68d26d085deede43915bfc7efbdc'
PY_PIECES = ('EXP_WINDOW', 'MAX_ESCAPE_DENOM', 'window_base', '_LAYER_THREADS', 'pack_cgemv', 'cgemv_threads')
STR_PIECES = ('GEMV_CPP_DECLS', 'GEMV_CUDA_SRC')
RENAME = {'EXP_WINDOW': 'CG_EXP_WINDOW', 'MAX_ESCAPE_DENOM': 'CG_MAX_ESCAPE_DENOM', 'window_base': 'cg_window_base',
          '_LAYER_THREADS': 'CG_LAYER_THREADS', 'pack_cgemv': 'cg_pack', 'cgemv_threads': 'cg_threads',
          'GEMV_CPP_DECLS': 'CG_CPP_DECLS', 'GEMV_CUDA_SRC': 'CG_CUDA_SRC'}

BLOCK = r'''

# C2 m = 1 GEMVs (muse_bf16_proj 3044ee7, public 9eba96894ee1402e), appended by tools/dl_cg_patch.py.
@@PIECES@@

CG_LAYERS = ("qkv_gate", "o", "unpooler")
CG_SLICE = 1024   # the unpooler GEMV at m <= 8 serves the first 1024 rows
CG_ON = True
CG_GATE_REL = 2e-3   # cgemv against the release's own sk2 output on the same input
_CG_EXT = None
_CG_ERR = []
_CG_THREAD = None
import hashlib as _cg_hashlib


def _cg_build():
    global _CG_EXT
    try:
        kw = dict(name="okc_dl_cgemv_" + _cg_hashlib.md5(CG_CUDA_SRC.encode()).hexdigest()[:12],
                  cpp_sources="#include <torch/extension.h>\n" + CG_CPP_DECLS, cuda_sources=CG_CUDA_SRC,
                  functions=["cgemv_fwd"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                  verbose=False)
        try:
            _CG_EXT = load_inline(no_implicit_headers=True, **kw)
        except TypeError:
            kw["cuda_sources"] = CG_CUDA_SRC.replace("#include <ATen/core/Tensor.h>", "#include <torch/types.h>")
            _CG_EXT = load_inline(**kw)
    except Exception as ex:  # noqa: BLE001
        _CG_ERR.append(ex)


def _cg_call(e, x, k):
    out = torch.empty((1, e["n"]), dtype=torch.bfloat16, device=x.device)
    _CG_EXT.cgemv_fwd(e["stream"], e["table"], e["pmax"], e["base4"], x.view(1, k), out, e["nth"])
    return out


_cg_prior_sk2_prepare = bfp_sk2_prepare
def bfp_sk2_prepare(weights, state):
    _cg_prior_sk2_prepare(weights, state)
    state["cg"] = {}
    if not (CG_ON and state.get("sm90")):
        return
    if _CG_THREAD is not None:
        _CG_THREAD.join()
    if _CG_EXT is None:
        print("CG disabled:", repr(_CG_ERR[0])[:200] if _CG_ERR else "no extension", flush=True)
        return
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(97531)
    for layer in CG_LAYERS:
        w = weights[layer][:CG_SLICE] if layer == "unpooler" else weights[layer]
        w = w.contiguous()
        n, k = w.shape
        try:
            p = cg_pack(w)
            if p is None:
                print("CG", layer, "too many escapes", flush=True)
                continue
            e = {"stream": p["stream"], "table": p["table"], "pmax": p["pmax"], "base4": p["base4"], "n": n,
                 "nth": cg_threads(n, k)}
            x = torch.randn(1, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
            a = _cg_call(e, x, k).double()
            b = bfp__run_sk2(x, layer, 1, n, k, state).double()
            torch.cuda.synchronize()
            rel_ab = ((a - b).norm() / b.norm()).item()
            rel_a, rel_b = ((a - ref).norm() / ref.norm()).item(), ((b - ref).norm() / ref.norm()).item()
            ok = rel_ab < CG_GATE_REL and rel_a < 5e-3
            print(f"CG_GATE {layer} cg_vs_sk2 {rel_ab:.2e} cg_vs_fp64 {rel_a:.2e} sk2_vs_fp64 {rel_b:.2e} "
                  f"equal {int((a == b).sum())}/{n} {'ok' if ok else 'DROP'}", flush=True)
            if ok:
                state["cg"][layer] = e
        except Exception as ex:  # noqa: BLE001
            print("CG", layer, "failed:", repr(ex)[:200], flush=True)


_cg_prior_candidates = bfp__candidates
def bfp__candidates(layer, m, state):
    c = _cg_prior_candidates(layer, m, state)
    if m == 1 and layer in state.get("cg", {}) and bfp__route_key(layer, m, "cg") not in state.get("disabled", ()):
        c.insert(0, ("cg",))
    return c


_cg_prior_run = bfp__run
def bfp__run(choice, x, layer, m, n, k, state):
    if choice[0] == "cg":
        e = state.get("cg", {}).get(layer)
        if (e is not None and m == 1 and n == e["n"] and x.dtype == torch.bfloat16 and x.is_contiguous()
                and x.numel() == k and x.data_ptr() % 16 == 0):
            return _cg_call(e, x, k)
        return _cg_prior_run(bfp__static_choice(layer, m, state), x, layer, m, n, k, state)
    return _cg_prior_run(choice, x, layer, m, n, k, state)


_cg_prior_bfp_prepare = bfp_prepare
def bfp_prepare(weights):
    state = _cg_prior_bfp_prepare(weights)
    for layer, m, v in state.get("tune_log", []):
        if m == 1 and layer in CG_LAYERS:
            print("CG_TUNE", layer, v, "->", state.get("choice", {}).get((layer, 1)), flush=True)
    return state


_cg_prior_prepare = prepare
def prepare(weights):
    global _CG_THREAD
    if CG_ON and _CG_EXT is None and _CG_THREAD is None:
        _CG_THREAD = threading.Thread(target=_cg_build, daemon=True)
        _CG_THREAD.start()
    state = _cg_prior_prepare(weights)
    mp = state.get("mm_pick", {})
    print("CG_CHAIN", {nm: mp.get((1, nm)) for nm in ("qkv", "o", "un")},
          {ly: (state.get("bfp") or {}).get("choice", {}).get((ly, 1)) for ly in CG_LAYERS}, flush=True)
    return state
'''


def pieces(c2):
    tree = ast.parse(c2)
    lines = c2.split('\n')
    got = {}
    for node in tree.body:
        names = []
        if isinstance(node, ast.FunctionDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for nm in names:
            if nm in PY_PIECES + STR_PIECES:
                if nm in got:
                    raise ValueError(f'{nm} defined twice in C2')
                got[nm] = '\n'.join(lines[node.lineno - 1:node.end_lineno])
    missing = set(PY_PIECES + STR_PIECES) - set(got)
    if missing:
        raise ValueError(f'C2 pieces missing: {sorted(missing)}')
    out = []
    for nm in PY_PIECES:
        text = got[nm]
        for a, b in RENAME.items():
            text = re.sub(r'\b' + a + r'\b', b, text)
        out.append(text)
    for nm in STR_PIECES:
        head, rest = got[nm].split('=', 1)
        out.append(RENAME[nm] + ' =' + rest)
    return '\n\n'.join(out)


def build(release, c2):
    if hashlib.sha256(release).hexdigest() != RELEASE_SHA:
        raise ValueError('release bytes changed')
    if hashlib.sha256(c2).hexdigest() != C2_SHA:
        raise ValueError('C2 bytes changed')
    src, c2s = release.decode(), c2.decode()
    for a in list(RENAME.values()) + ['_cg_prior', '_CG_EXT', 'CG_LAYERS']:
        if re.search(r'\b' + a, src):
            raise ValueError(f'{a} already in the release')
    for need in ('def bfp_sk2_prepare(', 'def bfp__candidates(', 'def bfp__run(', 'def bfp_prepare(', 'def prepare(',
                 'import threading', 'from torch.utils.cpp_extension import load_inline',
                 'def bfp__run_sk2(', 'def bfp__route_key(', 'def bfp__static_choice('):
        if need not in src:
            raise ValueError(f'release lacks {need}')
    block = BLOCK.replace('@@PIECES@@', pieces(c2s))
    return (src.rstrip('\n') + '\n' + block).encode()


def main():
    rel, c2, out = sys.argv[1:4]
    data = build(open(rel, 'rb').read(), open(c2, 'rb').read())
    ast.parse(data.decode())
    open(out, 'wb').write(data)
    print(out, hashlib.sha256(data).hexdigest(), len(data))


if __name__ == '__main__':
    main()
