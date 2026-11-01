"""Standalone-only owned51/duplicate vs Humming MASKED/duplicate diagnostic.

One owned prepare/extension compilation; immutable weights shared, scratch and locks
independent. Official judged path remains the unchanged owned kernel. No donor code.
"""
import ast
from pathlib import Path
import sys
from prefill_humming_indexed import BODY
from prefill_humming_masked import MASKED
from moe_make_bench_ab import AB

BENCH=r'''
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum
from humming.forward import humming_forward as _mh_forward
import random as _mh_random
import math

class _MHView:
    def __init__(self,layer):
        self.humming_config=layer.humming_config
        self.weight=layer.weight
        self.weight_scale=layer.weight_scale
        self.weight_scale_2=layer.weight_scale_2
        self.locks=torch.zeros_like(layer.locks)
    def __call__(self,x,**kw):
        return _mh_forward(self.humming_config,inputs=x,weight=self.weight,
            weight_scale=self.weight_scale,weight_scale_2=self.weight_scale_2,
            locks=self.locks,**kw)

_mh_owned_prepare=prepare
def prepare(weights):
    began=time.monotonic()
    base=_mh_owned_prepare(weights)
    logs=[]
    def emit(line):
        logs.append(line); print(line,flush=True)
    def guard():
        if time.monotonic()-began>280:
            raise RuntimeError("Standalone masked comparison incomplete: 280s guard")
    guard()
    dev=weights["w13"].device
    local={k:weights[k][:32].contiguous() for k in
           ("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")}
    hum=_hi_prepare(local)
    duplicate=dict(base,scratch=_scratch(dev))
    norm=(weights["post_norm_w"].float()+1).to(torch.bfloat16).contiguous()
    states={"owned":base,"owned2":duplicate,
        "masked":{"hum":hum,"post_expert_w":norm},
        "masked2":{"hum":{"layers":[_MHView(l) for l in hum["layers"]],"workspace":{}},
                   "post_expert_w":norm}}
    assert 51 in MAIN_MODES
    mode=VARIANT_MAIN0+MAIN_MODES.index(51)
    assert mode not in base.get("main_bad",set()),"owned51 failed self test"
    def run(tag,data):
        if tag.startswith("owned"): return _run(data,states[tag],mode)
        assert data["config"]["local_start"]==0
        return _mh_experts(data["hidden_states"],data["topk_weights"],data["topk_ids"],states[tag])
    flush=torch.empty((256<<20,),device=dev,dtype=torch.uint8)
@@ROUTE@@
    for m in (128,256):
        for routing in ("layer","std","skewed"):
            for seed in ((291,) if routing=="skewed" else (291,292)):
                guard()
                data=inputs(m,routing,seed)
                ref=_ref_forward(data,weights)
                graph={}; outputs={}; events={}
                for tag in states:
                    val=run(tag,data)
                    rel=_rel(val,ref)
                    scale=ref.float().norm(dim=-1).clamp(min=.1*ref.float().norm(dim=-1).mean())
                    row=float(((val.float()-ref.float()).norm(dim=-1)/scale.clamp(min=1e-12)).max())
                    assert bool(torch.isfinite(val).all()) and rel<TEST_TOL and row<.075, f"MH correctness {m} {routing} {seed} {tag} rel{rel} row{row}"
                    emit(f"MHCHECK M={m} route={routing} seed={seed} arm={tag} rel={rel:.6g} row={row:.6g}")
                    if routing!="skewed":
                        graph[tag]=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph[tag]): outputs[tag]=run(tag,data)
                        events[tag]=[]
                if routing=="skewed": continue
                for round in range(20):
                    order=list(states); _mh_random.Random(seed+round).shuffle(order)
                    for tag in order:
                        flush.zero_()
                        a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
                        a.record(); graph[tag].replay(); z.record(); events[tag].append((a,z))
                torch.cuda.synchronize()
                times={tag:[a.elapsed_time(z)*1000 for a,z in pairs] for tag,pairs in events.items()}
                med=lambda v:sorted(v)[len(v)//2]
                raw={tag:med(t) for tag,t in times.items()}
                paired={tag:med([a-b for a,b in zip(t,times["owned"])]) for tag,t in times.items()}
                emit(f"MHTIME M={m} route={routing} seed={seed} raw_us={raw}")
                emit(f"MHPAIR M={m} route={routing} seed={seed} vsowned_us={paired}")
                del graph,outputs,ref
    guard()
    print("MHRESULT complete requested128/256 layer/std two-seed +skew correctness",flush=True)
    for line in logs: print(line,flush=True)
    return base
'''


def build(src):
    tree=ast.parse(BODY)
    chunks=[]
    lines=BODY.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)) or (isinstance(node,ast.FunctionDef) and node.name in ('_hi_rank','_hi_prepare')):
            start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
            chunks.append(''.join(lines[start-1:node.end_lineno]))
    masked=MASKED.replace('_hi_workspace','_mh_workspace').replace('def _experts(', 'def _mh_experts(')
    route=AB[AB.index("    # The layer's judge routing"):AB.index('    pubfx = []')]
    start=route.index('    best = None')
    end=route.index('    def inputs(',start)
    route=route[:start]+'    rt=router(8)  # previously calibrated fixed route; no seed search\n'+route[end:]
    out=src+'\n'+'\n\n'.join(chunks)+'\n'+masked+'\n'+BENCH.replace('@@ROUTE@@',route)
    ast.parse(out)
    return out


if __name__=='__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text()))
