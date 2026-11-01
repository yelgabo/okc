"""Independent zero-job repair for the pinned G1b2 device literal; no GPU work.

Only wgs_stage_at changes. The caller keeps all joins, prefix initialization,
grid barriers and normal finalization. A changed upstream literal fails closed.
"""
import ast
import hashlib
from pathlib import Path
import sys

DEVICE_PIN = '1540fe13d00083d90f9e51f84b6002d18c51c41a4d8259197cb7d62d73f76215'
HEAD = '__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {\n'
GUARD = '  if (nj == 0) return 0;  // Empty routing: no initialized job metadata exists.\n'


def device_node(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'MOE_CUDA_SRC' for t in n.targets)]
    assert len(nodes) == 1, 'Expected exactly one MOE_CUDA_SRC assignment'
    node = nodes[0]
    assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    return node, node.value.value


def patch_empty_routing(source):
    node, device = device_node(source)
    assert hashlib.sha256(device.encode()).hexdigest() == DEVICE_PIN, 'Unreviewed device source'
    assert device.count(HEAD) == 1 and GUARD not in device
    # Guard before even forming the metadata pointer, for both range APIs.
    changed = device.replace(HEAD, HEAD + GUARD)
    assert "'''" not in changed
    lines = source.splitlines(keepends=True)
    value = node.value
    # Column offsets are UTF-8 bytes. This pinned assignment is ASCII.
    assert lines[value.lineno-1].isascii() and lines[value.end_lineno-1].isascii()
    prefix = ''.join(lines[:value.lineno-1]) + lines[value.lineno-1][:value.col_offset]
    suffix = lines[value.end_lineno-1][value.end_col_offset:] + ''.join(lines[value.end_lineno:])
    result = prefix + "r'''" + changed + "'''" + suffix
    before, after = ast.parse(source), ast.parse(result)
    a = next(n for n in before.body if isinstance(n, ast.Assign) and n.lineno == node.lineno)
    b = next(n for n in after.body if isinstance(n, ast.Assign) and n.lineno == node.lineno)
    _, actual = device_node(result)
    assert actual == changed
    a.value = ast.Constant('DEVICE_LITERAL'); b.value = ast.Constant('DEVICE_LITERAL')
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    return result


if __name__ == '__main__':
    src, dst = map(Path, sys.argv[1:])
    out = patch_empty_routing(src.read_text(encoding='utf-8'))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(out, encoding='utf-8', newline='\n')
    print('EMPTY_PATCH', len(out.encode()), hashlib.sha256(out.encode()).hexdigest())
