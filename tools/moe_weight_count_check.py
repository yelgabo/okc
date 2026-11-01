"""Generate count-only attribution check; official owned kernel stays unchanged.

The CPU map is embedded at generation time. No runtime file reads, new CUDA kernel,
copy probe, production dispatch change, or physical-traffic claim.
"""
import ast
from pathlib import Path
import sys
from moe_make_bench_ab import AB

COUNTS=r'''
import math
@@MODEL@@

_wc_original_prepare=prepare
def prepare(weights):
    began=time.monotonic()
    state=_wc_original_prepare(weights)
    dev=weights["w13"].device
    g_actual=torch.cuda.get_device_properties(dev).multi_processor_count
    assert g_actual%2==0,"count model requires even two-CTA grouping"
@@ROUTE@@
    logs=[]
    passed=True
    for m in (16,32):
        for routing in ("layer","std"):
            for seed in (311,312):
                if time.monotonic()-began>280:
                    raise RuntimeError("Count attribution incomplete: 280s guard")
                data=inputs(m,routing,seed)
                counts=torch.bincount(data["topk_ids"].flatten().long(),minlength=256)[:32].cpu().tolist()
                stats=summary(counts,g_actual//2)
                screen=stats["paired_saved_request_fraction"]>=.10
                passed=passed and screen
                logs.append(f"WCOUNT M={m} routing={routing} seed={seed} Gactual={g_actual} screen10pct={screen} stats={stats}")
    print("WCOUNTRESULT complete8cases fixedlayerrouter8; logicalrequestsONLY HBM/L2unmeasured",flush=True)
    for line in logs: print(line,flush=True)
    print(f"WCOUNTSCREEN all8_atleast10pct={passed}",flush=True)
    return state
'''


def build(src,model):
    tree=ast.parse(model)
    main=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
              and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')
    model=''.join(model.splitlines(keepends=True)[:main.lineno-1])
    route=AB[AB.index("    # The layer's judge routing"):AB.index('    pubfx = []')]
    a=route.index('    best = None'); z=route.index('    def inputs(',a)
    route=route[:a]+'    rt=router(8)  # fixed previously calibrated router; no seed search\n'+route[z:]
    body=COUNTS.replace('@@MODEL@@',model).replace('@@ROUTE@@',route)
    out=src+'\n'+body
    ast.parse(out)
    return out


if __name__=='__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text(),
                                     Path(__file__).with_name('moe_weight_reuse_model.py').read_text()))
