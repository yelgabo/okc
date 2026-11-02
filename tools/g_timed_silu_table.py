"""One fixed fully timed WGS BF16 SiLU table. No GPU or judge actions."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import time
import types
from pathlib import Path
from prod_size_compile_g import bindings

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
PIN = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
NATIVE_PIN = 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71'
EVIDENCE = ROOT / 'docs/steering/evidence/2026-11-02-worker-timed-silu-table'
PRODUCER = '''__global__ void timed_silu_table_build(float* table) {
  const unsigned raw = blockIdx.x * blockDim.x + threadIdx.x;
  const float gate = __uint_as_float(raw << 16);
  table[raw] = SILU_DIV(gate, 1.f + __expf(-gate));
}

'''
BINDING = '''  TORCH_CHECK((scratch.size() == 4 || scratch.size() == 5) && scratch.back().is_cuda()
              && scratch.back().device() == x.device() && scratch.back().is_contiguous()
              && scratch.back().scalar_type() == torch::kFloat32 && scratch.back().numel() == 65536,
              "TIMED_SILU_TABLE caller-owned scratch ABI");
  p.silu_table = (const float*)scratch.back().data_ptr();
'''
LAUNCH = '''  okc::timed_silu_table_build<<<256, 256, 0, st>>>((float*)scratch.back().data_ptr());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TIMED_SILU_TABLE producer launch");
'''
ALLOCATION = 'torch.empty(65536, dtype=torch.float32, device=dev)'
OLD_GATE = '        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;'
NEW_GATE = '        const float v = __ldg(p.silu_table + (__float_as_uint(gt) >> 16)) * up;'


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def once(source, old, new):
    assert source.count(old) == 1, ('nonunique intended edit', old, source.count(old))
    return source.replace(old, new)


def function(source, name):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    return ast.get_source_segment(source, nodes[-1])


def edits(source):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                and ast.unparse(node.targets[0]) == 'MOE_CUDA_SRC')
    old = ast.get_source_segment(source, node)
    assert sha(ast.literal_eval(node.value)) == NATIVE_PIN
    replacements = [
        ('struct Params {\n', 'struct Params {\n  const float* silu_table;\n'),
        (OLD_GATE, NEW_GATE),
        ('}  // namespace okc', PRODUCER + '}  // namespace okc'),
        ('  okc::Params p;\n', '  okc::Params p;\n' + BINDING),
        ('  void* args[]={&p};\n', LAUNCH + '  void* args[]={&p};\n'),
        ('  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;',
         '  p.dbg = scratch.size() > 4 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;'),
    ]
    updated = old
    for before, after in replacements:
        updated = once(updated, before, after)
    factory = function(source, '_ma_make_scratch')
    assert factory.endswith(']')
    new_factory = factory[:-1] + ',\n            ' + ALLOCATION + ']'
    scratch = '    st["ma_scratch"] = [_guarded(t) for t in _ma_make_scratch(dev)]\n'
    new_scratch = '    buffers = _ma_make_scratch(dev)\n    st["ma_scratch"] = [_guarded(t) for t in buffers[:3]] + [buffers[3]]\n'
    startup = "    private['ma_scratch'], tails = _release_private_scratch(torch, scratch_factory(device), sentinel)"
    new_startup = "    buffers = scratch_factory(device)\n    private['ma_scratch'], tails = _release_private_scratch(torch, buffers[:3], sentinel)\n    private['ma_scratch'].append(buffers[3])"
    return [(old, updated), (factory, new_factory), (scratch, new_scratch), (startup, new_startup)]


def build(source):
    assert sha(source) == PIN and '\r' not in source, 'exact physical LF public607 required'
    result = source
    ledger = edits(source)
    for old, new in ledger:
        result = once(result, old, new)
    reverse = result
    for old, new in reversed(ledger):
        reverse = once(reverse, new, old)
    assert reverse == source
    assert len(result.encode()) < 1_000_000 and '\r' not in result
    ast.parse(result)
    return result, dict(base_sha256=PIN, source_sha256=sha(result), bytes=len(result.encode()),
                        exact_reverse=True, edits=4, selected_native_sha256=sha(bindings(result)['MOE_CUDA_SRC']),
                        scope='WGS B4/MAIN only; full 65536 entries on each timed native call; short paths unchanged',
                        CUDA_numeric_equivalence='NOT RUN', GPU=False, performance_qualified=False)


def validate(source, original):
    expected, manifest = build(original)
    assert source == expected, 'source contract differs from the single fixed treatment'
    base_env, env = bindings(original), bindings(source)
    assert env['_MA_ALT_SRCS'] == base_env['_MA_ALT_SRCS'], 'short/helper source changed'
    assert env['MOE_CPP_SRC'] == base_env['MOE_CPP_SRC'], 'host vector ABI changed'
    for name in ('kernel_fn', '_ma_ext', '_fork_native_moe'):
        assert function(source, name) == function(original, name), name
    native = env['MOE_CUDA_SRC']
    assert native.count(NEW_GATE) == 1 and native.count(PRODUCER) == 1
    assert native.count('SILU_DIV(gt, 1.f + __expf(-gt))') == 4
    assert native.count(LAUNCH) == 1
    assert native.index(LAUNCH) < native.index('const cudaError_t launch=cudaLaunchCooperativeKernel')
    assert 'const float gt = fork_bf16(d4[h] * gsc), up = fork_bf16(d4[2 + h] * gsc);' in native
    return manifest


def generate(out):
    original = BASE.read_bytes().decode()
    source, manifest = build(original)
    validate(source, original)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'candidate.py').write_bytes(source.encode())
    (out / 'selected.cu').write_bytes(bindings(source)['MOE_CUDA_SRC'].encode())
    (out / 'generation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return source, manifest


def compile_fixed(out):
    from test_g_timed_silu_table import capture, coverage
    source, generation = generate(out / 'generated')
    calls, routes = capture(source)
    originals, old_routes = capture(BASE.read_bytes().decode())
    assert routes == old_routes
    driver_path = ROOT / '.build/research-gamma13/artifacts/adapted-driver.py'
    driver = driver_path.read_text()
    assert sha(driver) == 'e695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c'
    old = "expressions = [b'&okc::moe_fused<false>', b'&okc::moe_fused<true>']"
    driver = once(driver, old, old[:-1] + ", b'&okc::timed_silu_table_build']")
    namespace = dict(__name__='_timed_silu_fixed_secondary13')
    exec(compile(driver, str(driver_path), 'exec'), namespace)
    result = dict(started_epoch=time.time(), generation=generation, routes=routes, records=[],
                  decision='RUNNING_FIXED_SECONDARY_SCREEN', GPU=False,
                  scope='NVRTC13/PTXAS13 fixed exact cached controls; host NVCC/link and CUDA numerics NOT RUN',
                  driver_sha256=sha(driver_path.read_bytes()), adapted_driver_sha256=sha(driver),
                  coverage={name: coverage(native) for name, native in [('MAIN_original', bindings(BASE.read_bytes().decode())['MOE_CUDA_SRC']), ('MAIN_treatment', bindings(source)['MOE_CUDA_SRC']), ('helper_unchanged', bindings(source)['_MA_ALT_SRCS']['moe_helper0'])]})
    summary = out / 'compile-result.json'
    def save():
        summary.write_text(json.dumps(result, indent=2) + '\n')
    factory_rows = []
    for index, (original, call) in enumerate(zip(originals, calls)):
        factory_rows.append(dict(index=index, historical_short_RMW_ON=index >= 5,
            actual_flags=call['extra_cuda_cflags'], original_name=original['name'], treatment_name=call['name'],
            original_cuda_sha256=sha(original['cuda_sources']), treatment_cuda_sha256=sha(call['cuda_sources']),
            cpp_equal=original['cpp_sources'] == call['cpp_sources'],
            flags_equal=original['extra_cuda_cflags'] == call['extra_cuda_cflags'],
            unchanged=original == call))
    result['factory_capture'] = factory_rows
    save()
    for index, family in [(3, 'B4'), (4, 'MAIN')]:
        call = calls[index]
        control_dir = ROOT / '.build/worker-g224-reference' / family
        control = json.loads((control_dir / 'manifest.json').read_text())
        assert control['source_sha256'] == NATIVE_PIN
        assert sha((control_dir / 'original-selected.cu').read_bytes()) == NATIVE_PIN
        assert sha((control_dir / 'device.ptx').read_bytes()) == control['ptx_sha256']
        assert sha((control_dir / 'device.cubin').read_bytes()) == control['cubin_sha256']
        folder = out / family
        folder.mkdir(exist_ok=True)
        native_path = folder / 'selected.cu'
        native_path.write_bytes(call['cuda_sources'].encode())
        record = dict(family=family, control_manifest=control,
                      actual_flags=call['extra_cuda_cflags'], control_dir=str(control_dir))
        result['records'].append(record)
        save()
        args = types.SimpleNamespace(source=native_path, out=folder, cache=Path('/root/.cache/okc-cuda130-88'),
                    sc6=1, define=[flag for flag in call['extra_cuda_cflags'] if flag.startswith('-D')], forward_declare_wgs=True)
        try:
            namespace['compile_source'](args)
            manifest = json.loads((folder / 'manifest.json').read_text())
            record['treatment_manifest'] = manifest
            assert manifest['options'] == control['options'], 'unmatched cached options'
            assert manifest['virtual_headers'] == control['virtual_headers']
            assert manifest['declaration_adapter'] == control['declaration_adapter']
            disassembler = args.cache / 'cuda_nvdisasm-linux-x86_64-13.0.85-archive/bin/nvdisasm'
            disassembly = subprocess.run([str(disassembler), str(folder / 'device.cubin')], capture_output=True, text=True, check=True).stdout
            (folder / 'device.sass').write_text(disassembly)
            original_resources = control['resources']
            resources = manifest['resources']
            record['resource_deltas'] = {symbol: {key: resources[symbol][key] - values[key] for key in values}
                                         for symbol, values in original_resources.items()}
            record['resources_no_worse'] = all(delta <= 0 for row in record['resource_deltas'].values() for delta in row.values())
            record['bounds_equal'] = manifest['ptx_launch_bounds'] == control['ptx_launch_bounds']
            record['producer_resources'] = {symbol: value for symbol, value in resources.items() if symbol not in original_resources}
            record['producer_no_spills'] = all(value[key] == 0 for value in record['producer_resources'].values() for key in ('stack_bytes', 'spill_store_bytes', 'spill_load_bytes'))
            record['shared_layout_source_identical'] = call['cuda_sources'][call['cuda_sources'].index('struct Smem'):call['cuda_sources'].index('struct Smem') + 250] == originals[index]['cuda_sources'][originals[index]['cuda_sources'].index('struct Smem'):originals[index]['cuda_sources'].index('struct Smem') + 250]
            save()
            if not (record['resources_no_worse'] and record['bounds_equal'] and record['producer_no_spills']):
                result['decision'] = 'STOP_FIXED_TREATMENT_RESOURCE_GATE'
                result['finished_epoch'] = time.time()
                save()
                return result
        except Exception as error:
            result.update(decision='STOP_FIXED_COMPILE_OR_IDENTITY', blocker=repr(error), finished_epoch=time.time())
            save()
            raise
    result.update(decision='SECONDARY_SURVIVOR_NOT_TARGET_QUALIFIED', finished_epoch=time.time())
    save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EVIDENCE / 'generated')
    parser.add_argument('--compile-fixed', action='store_true')
    args = parser.parse_args()
    result = compile_fixed(args.out) if args.compile_fixed else generate(args.out)[1]
    print(json.dumps(result, indent=2))

