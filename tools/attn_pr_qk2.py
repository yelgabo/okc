"""Owned OFF-by-default two-chain q1 dot product, MODE11/13 only."""
import ast
import hashlib
from pathlib import Path

PIN = '6e29818bad3607a99b85c36d1a415f31e5dead5be8a95713653d837363188ef0'
OLD = '    constexpr bool QK2 = (ONE_F1 & 1) && FLAT && L == 1;'
NEW = '''    constexpr bool QK2 = ((ONE_F1 & 1) && FLAT && L == 1) ||
                         (ONE_PR_QK2 && L == 1 && (MODE == 11 || MODE == 13));'''
FLAG = '''#ifndef ONE_PR_QK2
#define ONE_PR_QK2 0  // Owned diagnostic: two Q.K chains, q1 MODE11/13 only.
#endif
'''


def patch_device(device):
    assert device.count(OLD) == 1 and 'ONE_PR_QK2' not in device
    anchor = '#ifndef ONE_F1\n'
    assert device.count(anchor) == 1
    return device.replace(anchor, FLAG + anchor).replace(OLD, NEW)


def _assignment(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def patch_source(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PIN, 'Use LF frozen891 control'
    node = _assignment(source, '_ONE_SRC')
    before = ast.literal_eval(node.value)
    after = patch_device(before)
    # Do not touch any other Python/CUDA route. The one literal's exact reverse
    # transform is part of the offline scope gate.
    assert after.replace(FLAG, '').replace(NEW, OLD) == before
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ['_ONE_SRC = ' + repr(after) + '\n']
    result = ''.join(lines)
    oldtree, newtree = ast.parse(source), ast.parse(result)
    _assignment_value = lambda tree: next(n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == '_ONE_SRC' for t in n.targets))
    _assignment_value(oldtree).value = ast.Constant('DEVICE')
    _assignment_value(newtree).value = ast.Constant('DEVICE')
    assert ast.dump(oldtree, include_attributes=False) == ast.dump(newtree, include_attributes=False)
    return result


RESOURCE = r'''
std::vector<int64_t> attn_q2_resources() {
    std::vector<int64_t> out;
#define Q2RES(L,M) { Instance<L,M>::prepare(); cudaFuncAttributes a = {}; \
    TORCH_CHECK(cudaFuncGetAttributes(&a, attn_one_kernel<L,M>)==cudaSuccess,"Q2 attrs"); \
    out.insert(out.end(), {L,M,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,Instance<L,M>::BYTES,Instance<L,M>::NT,Instance<L,M>::per_sm}); }
    ONE_INSTANCES(Q2RES)
#undef Q2RES
    return out;
}
'''


def resource_source(source):
    node = _assignment(source, '_ONE_SRC')
    device = ast.literal_eval(node.value)
    anchor = 'std::vector<int64_t> attn_one_prepare() {'
    assert device.count(anchor) == 1
    device = device.replace(anchor, RESOURCE + '\n' + anchor)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ['_ONE_SRC = ' + repr(device) + '\n']
    source = ''.join(lines)
    decl = 'std::vector<int64_t> attn_one_prepare();'
    assert source.count(decl) == 1
    source = source.replace(decl, decl + '\\nstd::vector<int64_t> attn_q2_resources();')
    exports = 'functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],'
    assert source.count(exports) == 1
    return source.replace(exports, 'functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop", "attn_q2_resources"],')


def build(control_path, output_path, harness_root):
    import importlib.util
    source = control_path.read_text(encoding='utf-8')
    patched = patch_source(source)
    # Only retained privately; release route defaults remain zero.
    Path(str(output_path) + '.device.cu').write_text(ast.literal_eval(_assignment(patched, '_ONE_SRC').value), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('_q2_helpers', harness_root / 'tools/attn_make_bench_one.py')
    helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
    runner = helpers.RUNNER.format(variants=['_one'], coop_probe=True)
    head, sep, _ = runner.rpartition('\nimport os as _os\n'); assert sep
    common = resource_source(patched)
    # The judged function remains the original control. Diagnostic
    # modules execute one embedded owned body under distinct extension names.
    body = (Path(__file__).parent / 'attn_pr_qk2_body.py').read_text(encoding='utf-8')
    result = ('import time as _q2_time\n_Q2_IMPORT_STARTED = _q2_time.monotonic()\n' + common
              + '\n_Q2_STARTED_SOURCE = ' + repr(common) + '\n' + head + '\n' + body)
    ast.parse(result)
    assert len(result.encode()) < 1_000_000
    output_path.write_text(result, encoding='utf-8', newline='\n')
    return len(result.encode()), hashlib.sha256(result.encode()).hexdigest()


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('control', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--harness-root', type=Path, required=True)
    a=p.parse_args(); print('Q2_BUILD', *build(a.control, a.output, a.harness_root))
