"""Port only existing standalone dense Lt pins into the selected decode layer.

Offline generator; no heuristic search, prefill route or changes to CUDA source.
The appended prepare gates every plan against full FP64 and fresh graph inputs.
"""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = {('unpooler', 128): (3, [66,18,1,0,0,2,35,0,5]),
        ('unpooler', 256): (1, [66,20,1,0,0,4,35,0,3]),
        ('qkv_gate', 256): (0, [66,124,1,0,0,0,35,0,2])}

RUNTIME = r'''
_dlplt_old_proj = _proj
_dlplt_old_prepare = prepare

def _proj(x, layer, n, state):
    entry = state.get('dlplt', {}).get((layer, x.shape[0]))
    if entry is not None and n == state['dense'][layer].shape[0]:
        if x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 256 == 0:
            return entry(x)
    return _dlplt_old_proj(x, layer, n, state)

def prepare(weights):
    state = _dlplt_old_prepare(weights)
    lt = _dlplt_build()
    state['dlplt'] = {}
    gen = torch.Generator(device=weights['qkv_gate'].device).manual_seed(1101443)
    report = []
    for (layer, m), (layout, cfg) in _DLPLT_PINS.items():
        w = state['dense'][layer]
        n, k = w.shape
        assert w.dtype == torch.bfloat16 and (n,k) == {
            'qkv_gate': (2304,8192), 'unpooler': (8192,4096)}[layer], (layer,w.shape,w.dtype)
        ws = torch.zeros(32 << 20, dtype=torch.uint8, device=w.device)
        warg = w.t().contiguous() if layout % 2 else w
        plan = lt.lt_plan_new(m,n,k,layout,ws.numel())
        idx = lt.lt_add_config(plan,list(cfg)) if plan >= 0 else -1
        assert idx >= 0, ('decode Lt pin rejected',layer,m,layout,cfg)
        def make(plan=plan,idx=idx,warg=warg,ws=ws,m=m,n=n):
            def run(x):
                out = torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
                rc = lt.lt_run(plan,idx,warg,x,out,ws)
                if rc: raise RuntimeError(f'decode Lt status {rc}')
                return out
            return run
        run = make()
        def check(x,run=run,w=w):
            out = run(x)
            ref = x.double() @ w.double().T
            delta = (out.double()-ref)
            rel = float(delta.norm()/ref.norm().clamp(min=1e-12))
            row = float((delta.norm(dim=-1)/ref.norm(dim=-1).clamp(min=1e-12)).max())
            assert bool(torch.isfinite(out).all()) and rel < .005 and row < .01, (layer,m,rel,row)
            return rel,row
        x = torch.randn((m,k),device=w.device,generator=gen).to(torch.bfloat16)
        err = check(x)
        probe = torch.zeros_like(x); probe[:,17] = 2
        assert check(probe) == (0.,0.)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured = run(x)
        for fresh in range(3):
            x.copy_(torch.randn(x.shape,device=w.device,generator=gen).to(torch.bfloat16))
            graph.replay(); torch.cuda.synchronize()
            ref = x.double() @ w.double().T
            assert bool(torch.isfinite(captured).all())
            assert float((captured.double()-ref).norm()/ref.norm()) < .005
        state['dlplt'][layer,m] = run
        report.append(f'DLPLT_GATE {layer} M{m} L{layout} cfg{cfg} info{list(lt.lt_info(plan,idx))} FP64rel,row{err} probe0 fresh3PASS')
    state['dlplt_report'] = report
    for line in report: print(line,flush=True)
    return state
'''

def compact_docstrings(source):
    """Remove only AST-designated Python docstrings to fit the paired payload."""
    tree=ast.parse(source); raw=source.encode(); lines=raw.splitlines(keepends=True)
    offsets=[0]
    for line in lines: offsets.append(offsets[-1]+len(line))
    edits=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
            first=node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                edits.append((offsets[first.lineno-1]+first.col_offset,
                              offsets[first.end_lineno-1]+first.end_col_offset))
    for a,z in sorted(edits,reverse=True): raw=raw[:a]+b'pass'+raw[z:]
    out=raw.decode();ast.parse(out)
    return out

def build(layer, projection):
    layer=compact_docstrings(layer)
    tree = ast.parse(projection)
    cpp = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t,ast.Name) and t.id == 'LT_CPP' for t in n.targets))
    funcs = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == 'LT_FUNCS' for t in n.targets))
    builder = next(ast.get_source_segment(projection,n) for n in tree.body
                   if isinstance(n,ast.FunctionDef) and n.name == '_build_lt')
    builder = builder.replace('_build_lt','_dlplt_build').replace('okc_lt_v2_','okc_dlplt_v1_')
    builder = builder.replace('cpp_sources=LT_CPP','cpp_sources=_DLPLT_CPP').replace('functions=LT_FUNCS','functions=_DLPLT_FUNCS')
    source = (layer + '\n_DLPLT_CPP = '+repr(cpp)+'\n_DLPLT_FUNCS = '+repr(funcs)
              +'\n_DLPLT_PINS = '+repr(PINS)+'\n'+builder+'\n'+RUNTIME)
    ast.parse(source)
    # The original layer is an exact prefix; all its CUDA, fences and policies survive.
    assert source.startswith(layer)
    assert len(source.encode()) < 1_000_000
    return source

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('layer',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    source=build(args.layer.read_text(encoding='utf-8'),
                 (ROOT/'muse_bf16_proj/submission.py').read_text(encoding='utf-8'))
    args.output.write_text(source,encoding='utf-8',newline='\n')
    print(len(source.encode()),hashlib.sha256(source.encode()).hexdigest())
