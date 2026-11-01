"""Generate a PRIVATE, CHECK-ONLY controlled standalone SC6 comparison.

No public submission or production incorporation of the attributed source.
"""
import ast
import hashlib
from pathlib import Path
import sys

from dl_fork_moe import patch_device, once
from dl_fork_moe_empty import HEAD, GUARD

PIN = '2d931f09133d058053dda833b51f1867ea06cdcac0c912521ad37e8f7853185c'
LF_PIN = '795ac642adc3370460ff04166ae19b5aac172a48aba0170b94dac2a6a3430410'
RESOURCE = r'''
std::vector<int64_t> sc6_resources() {
  std::vector<int64_t> out;
  const size_t sm=okc_smem_bytes();
  for (int big=0;big<2;++big) {
    auto k=big?okc::moe_fused<true>:okc::moe_fused<false>;
    cudaFuncAttributes a{}; int resident=0;
    TORCH_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)==cudaSuccess,"SC6 resource attribute");
    TORCH_CHECK(cudaFuncGetAttributes(&a,k)==cudaSuccess,"SC6 resource attrs");
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k,okc::NTL,sm)==cudaSuccess,"SC6 resource occupancy");
    for (int64_t x : {int64_t(a.numRegs),int64_t(a.localSizeBytes),int64_t(a.sharedSizeBytes),int64_t(sm),int64_t(resident),int64_t(okc::NTL)}) out.push_back(x);
  }
  return out;
}
'''


def make(source, baseline, harness):
    assert hashlib.sha256(source.encode()).hexdigest() == LF_PIN, 'Unreviewed standalone comparator'
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []
    found = set()
    for n in tree.body:
        if not isinstance(n, ast.Assign):
            continue
        name = ast.unparse(n.targets[0])
        if name not in ('CUDA_SRC', 'VAR_SRC', 'PS_SHAPES', 'PS_PATCHES'):
            continue
        if name in ('CUDA_SRC', 'VAR_SRC'):
            device = ast.literal_eval(n.value)
            device = patch_device(device, name == 'CUDA_SRC', wgs_down_sites=4 if name == 'CUDA_SRC' else None)
            if name == 'CUDA_SRC':
                device = once(device, HEAD, HEAD + GUARD)
            device += RESOURCE
            replacement = name + " = r'''" + device + "'''\n"
        else:
            replacement = name + ' = {}  # controlled comparison: no dedicated issuer source\n'
        found.add(name)
        edits.append((n.lineno-1, n.end_lineno, replacement))
    assert found == {'CUDA_SRC','VAR_SRC','PS_SHAPES','PS_PATCHES'}
    for a,z,text in sorted(edits,reverse=True):
        lines[a:z] = [text]
    result = ''.join(lines)
    # Disable the MID sentinel path too. TINY/TINY2 already explicitly disable it.
    result = once(result, 'MID_DEFINES += ("-DOKC_WSH0=1", "@helper0")',
                  'MID_DEFINES += ("-DOKC_DF=0", "-DOKC_WSH0=1", "@helper0")')
    # Materialize both read-only layouts from the SAME actual local weight slice.
    result = once(result, 'if MFAB and locals().get("sc6w", False):', 'if locals().get("sc6w", False):')
    result = once(result, 'std::vector<int64_t> okc_attrs();', 'std::vector<int64_t> okc_attrs();\nstd::vector<int64_t> sc6_resources();')
    result = once(result, 'functions=["okc_run", "okc_set_pdl", "okc_attrs"]', 'functions=["okc_run", "okc_set_pdl", "okc_attrs", "sc6_resources"]')
    result += '\n# CHECK ONLY: attributed controlled standalone comparison. Never submit.\n'
    result += '_SC_BASELINE = ' + repr(baseline) + '\n'
    result += harness
    ast.parse(result)
    assert len(result.encode()) < 1_000_000
    return result


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    src, dst = map(Path, sys.argv[1:])
    result = make(src.read_text(encoding='utf-8'),
                  (root/'muse_moe_w4a16/baseline.py').read_text(encoding='utf-8'),
                  (root/'tools/moe_sc6_compare_body.py').read_text(encoding='utf-8'))
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(result,encoding='utf-8',newline='\n')
    print('SC6_COMPARE',len(result.encode()),hashlib.sha256(result.encode()).hexdigest())
