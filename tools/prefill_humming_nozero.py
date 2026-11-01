"""Isolate indexed expert output clearing: zero/unfiltered vs nozero/filtered.

Input must be exact checked indexed-v3 candidate, with routed-M heuristic intact.
Generate PURE and four-arm BENCH files; no dispatch. Bench is never scored.
Proof: valid sorted routes form a bijection, so G1/G2 write all local rows.
Pinned SGLang a55ab7d activation.cuh skips -1 before input loads; post-expert
norm skips nonlocal rows before loads (T>=32), or torch.where before norm (<32).
Both stages and activation preserve original BF16 boundaries and fences.
Kernel boundaries use ordinary stream dependencies, PTX8.7 section8.9.4.
"""
import ast
from pathlib import Path
import sys

from prefill_humming_component_bench import BENCH


def build(src):
    old = '''    ws["c1"].zero_()
    ws["c2"].zero_()
    ws["c3"].zero_()'''
    assert src.count(old) == 1
    src = src.replace(old, '''    nozero = state["hum"].get("nozero", True)
    if not nozero:
        ws["c1"].zero_()
        ws["c2"].zero_()
        ws["c3"].zero_()''')
    old = '    silu_and_mul(ws["c1"],ws["c2"])'
    assert src.count(old) == 1
    src = src.replace(old, '''    if nozero:
        silu_and_mul(ws["c1"],ws["c2"],expert_ids=ws["local"],expert_step=1)
    else:
        silu_and_mul(ws["c1"],ws["c2"])''')
    src = src.replace('    # Humming writes only selected rows. Zero every other route so SiLU and\n'
                      '    # post-expert normalization never touch uninitialized nonlocal values.\n',
                      '    # Only local rows are consumed; filtered SiLU skips all other rows.\n')
    ast.parse(src)
    return src


def bench(src):
    b = BENCH
    old = '''    for tag in ("marlin","marlin2"):
        arms[tag]=dict(base,bank=_marlin_bank(weights))
'''
    assert b.count(old) == 1
    b = b.replace(old, '')
    b = b.replace('("R","R2","M","M2")', '("R","R2","Z","Z2")')
    b = b.replace('"shape_estimate":"local" if tag.startswith("M") else "routed"',
                  '"shape_estimate":"routed","nozero":tag.startswith("Z")')
    b = b.replace('(_hb_marlin_experts if tag.startswith("marlin") else _experts)', '_experts')
    b = b.replace('ref=run("marlin",x,tw,ids).clone()', 'ref=run("R",x,tw,ids).clone()')
    anchor = '                eager=run(tag,x,tw,ids)'
    assert b.count(anchor) == 1
    b = b.replace(anchor, '''                # Warm scratch/config, then poison every output before correctness.
                run(tag,x,tw,ids)
                for key in ("c1","c2","c3"):
                    arms[tag]["hum"]["workspace"][m][key].fill_(float("nan"))
                eager=run(tag,x,tw,ids)''')
    b = b.replace('@@SIZES@@', '(753,3012,16384)')
    src += '\n' + b
    ast.parse(src)
    return src


if __name__ == '__main__':
    assert len(sys.argv) in (4, 5)
    inp, pure, paired = map(Path, sys.argv[1:4])
    original = inp.read_text()
    if len(sys.argv) == 5:
        assert sys.argv[4] == '--nostream'
        anchor = '        ws={"rank":torch.empty((r,),dtype=torch.int32,device=dev),'
        assert original.count(anchor) == 1
        original = original.replace(anchor, '        for config in cfg: config["use_stream_k"]=False\n'+anchor)
    src = build(original)
    pure.write_text(src)
    paired.write_text(bench(src))
