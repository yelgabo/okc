"""Fresh replay gate at both changed heuristic sizes and incumbent M32 control."""
import ast
from pathlib import Path
import sys
from prefill_humming_stress import STRESS


def build(src):
    # Reuse unchanged numeric envelope, banks, 64 controls and 2000 queued replays.
    # Only shape-valued 32 literals change; expert count/ID bounds stay 32.
    body=STRESS.replace('_hi_plain_prepare = prepare\ndef prepare(weights):',
                        '_hi_plain_prepare = prepare\ndef _hi_stress_one(weights, m, state):')
    body=body.replace('    state=_hi_plain_prepare(weights)\n','')
    body=body.replace('(32,4096)', '(m,4096)').replace('.expand(32,8)', '.expand(m,8)')
    body=body.replace('(32,8)', '(m,8)').replace('"workspace"][32]', '"workspace"][m]')
    body=body.replace('"HUMFRESH M32 replays2000', 'f"HUMFRESH M{m} replays2000')
    body+='\ndef prepare(weights):\n    state=_hi_plain_prepare(weights)\n    for m in (32,753,3012):\n        _hi_stress_one(weights,m,state)\n    return state\n'
    out=src+body
    ast.parse(out)
    return out


if __name__=='__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text()))
