"""Poisoned grouped-capacity fresh replay gate; no benchmark core changes."""
import ast
from pathlib import Path
import sys
from prefill_humming_bounded_stress import build as three_shape_stress


def build(src):
    out=three_shape_stress(src)
    out=out.replace('_hi_plain_prepare = prepare',
                    'import time as _hm_clock\n_hi_plain_prepare = prepare')
    anchor_prepare='def prepare(weights):\n    state=_hi_plain_prepare(weights)\n    for m in (32,753,3012):'
    assert out.count(anchor_prepare)==1
    out=out.replace(anchor_prepare,'''def prepare(weights):
    started=_hm_clock.monotonic()
    state=_hi_plain_prepare(weights)
    for m in (32,753,3012):
        if _hm_clock.monotonic()-started>230:
            raise RuntimeError("Masked fresh gate incomplete: setup runtime guard")''')
    start=out.index('        for meta in ws["meta"]:')
    end=out.index('        if i%4==0:',start)
    out=out[:start]+'''        invalid=torch.maximum(invalid,(ws["counts"]>ws["cap"]).any().float())
'''+out[end:]
    anchor='        ref=_experts(x,tw,ids,state).clone()'
    assert out.count(anchor)==1
    out=out.replace(anchor,'''        poison_ws=_hi_workspace(x,state)
        for key in ("gx","c1","c2","c3"):
            poison_ws[key].fill_(float("nan"))
'''+anchor)
    # Every reference begins with poisoned grouped storage, and kernels must write
    # every valid row/tail. Queued changing-bank tests then check retained state.
    ast.parse(out)
    return out


if __name__=='__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text()))
