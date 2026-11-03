"""Generate a private standalone MoE attribution check from the pinned public source."""
import argparse
import ast
import inspect
from pathlib import Path

import dl_g1_wgs_frontier as frontier
from moe_make_sc6_compare import make

ROOT = Path(__file__).resolve().parents[1]


def build(source):
    baseline = (ROOT/'muse_moe_w4a16/baseline.py').read_text(encoding='utf-8')
    base = make(source, baseline, '')
    assign = next(n for n in ast.parse(base).body if isinstance(n,ast.Assign)
                  and getattr(n.targets[0],'id','') == 'CUDA_SRC')
    cuda = frontier.patch_device(ast.literal_eval(assign.value))
    assert "'''" not in cuda
    lines = base.splitlines(keepends=True)
    lines[assign.lineno-1:assign.end_lineno] = ["CUDA_SRC = r'''" + cuda + "'''\n"]
    checks = (ROOT/'tools/moe_sc6_compare_body.py').read_text(encoding='utf-8')
    error_fn = next(n for n in ast.parse(checks).body if isinstance(n,ast.FunctionDef) and n.name=='_sc_errors')
    helpers = [f'WARPS={frontier.WARPS}\nEXPERTS={frontier.EXPERTS}', inspect.getsource(frontier.trace_words),
               inspect.getsource(frontier.analyze), inspect.getsource(frontier.encode_trace),
               ast.get_source_segment(checks,error_fn)]
    result = ''.join(lines) + '\n' + '\n\n'.join(helpers) + '\n'
    result += (ROOT/'tools/moe_frontier_body.py').read_text(encoding='utf-8')
    compile(result,'<private MoE frontier check>','exec')
    if len(result.encode()) >= 1_000_000:
        raise ValueError('frontier check exceeds payload limit')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    args.output.write_bytes(build(args.input.read_text(encoding='utf-8')).encode())
