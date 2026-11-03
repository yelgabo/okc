"""Fixed four-entry NVRTC13/PTXAS13 screen for one chunk-phased treatment."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types

from g_expert_g224_chunk_phased import ROOT, generate
from g_expert_g224_prefix_nvrtc_layout import adapt_driver, adapt_device, inspect_ptx, PROBES
from test_g_expert_g224_chunk_phased import capture_treatment, contract


def run(out):
    variant, native, generated = generate(ROOT, out / 'generated')
    assert contract(native)
    calls, _ = capture_treatment(variant, True)
    selected = [call for call in calls if '-DOKC_WGS=1' in call['extra_cuda_cflags']]
    assert len(selected) == 2
    driver_path = ROOT / '.build/research-gamma13/artifacts/adapted-driver.py'
    driver = driver_path.read_text()
    assert hashlib.sha256(driver.encode()).hexdigest() == 'e695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c'
    adapted = adapt_driver(driver)
    namespace = dict(__name__='_g224_chunk_fixed_secondary13', adapt_device=adapt_device, inspect_ptx=inspect_ptx, PROBES=PROBES)
    exec(compile(adapted, str(driver_path), 'exec'), namespace)
    result = dict(scope='Offline device-prefix NVRTC13/PTXAS13 only; same attested layout, virtual-header and declaration-only adapters. No host NVCC/link, disassembly, GPU, CUDA driver API, timing or target occupancy.',
                  generator=generated, driver_sha256=hashlib.sha256(driver.encode()).hexdigest(), adapted_driver_sha256=hashlib.sha256(adapted.encode()).hexdigest(), started_epoch=time.time(), records=[])
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'compile-result.json'
    cache = Path('/root/.cache/okc-cuda130-88')
    for call in selected:
        family = 'B4' if '-DOKC_B4FIN=1' in call['extra_cuda_cflags'] else 'MAIN'
        assert call['cuda_sources'] == native
        folder = out / family
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / 'selected.cu'
        source.write_bytes(native.encode())
        record = dict(family=family, actual_flags=call['extra_cuda_cflags'], actual_selected_source_sha256=hashlib.sha256(call['cuda_sources'].encode()).hexdigest())
        result['records'].append(record)
        args = types.SimpleNamespace(source=source, out=folder, cache=cache, sc6=1, define=[flag for flag in call['extra_cuda_cflags'] if flag.startswith('-D')], forward_declare_wgs=True)
        try:
            namespace['compile_source'](args)
            manifest = json.loads((folder / 'manifest.json').read_text())
            record['manifest'] = manifest
            assert len(manifest['resources']) == 2
            assert set(manifest['resources']) == set(manifest['lowered'])
            record['zero_stack_spills'] = all(row['stack_bytes'] == 0 and row['spill_store_bytes'] == 0 and row['spill_load_bytes'] == 0 for row in manifest['resources'].values())
            record['rounded_register_blocks'] = {symbol: ((16384 // (((row['registers'] * 32 + 255) // 256) * 256)) * 4) // 10 for symbol, row in manifest['resources'].items()}
            record['register_gate'] = all(blocks >= 2 for blocks in record['rounded_register_blocks'].values())
            assert manifest['ptx_launch_bounds'].count('.maxntid 320, 1, 1') == 2
            assert manifest['ptx_launch_bounds'].count('.minnctapersm 2') == 2
            record['resource_screen_pass'] = record['zero_stack_spills'] and record['register_gate']
        except Exception as error:
            record['error'] = repr(error)
            result['decision'] = 'STOP_COMPILER_OR_INSTRUMENT'
            summary.write_text(json.dumps(result, indent=2) + '\n')
            raise
        summary.write_text(json.dumps(result, indent=2) + '\n')
    result['decision'] = 'RESOURCE_SCREEN_SURVIVOR_NOT_TARGET_QUALIFIED' if all(record['resource_screen_pass'] for record in result['records']) else 'STOP_FIXED_CHUNK_PHASED_RESOURCES'
    result['finished_epoch'] = time.time()
    summary.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(decision=result['decision'], rows=[dict(family=record['family'], resources=record['manifest']['resources']) for record in result['records']]), indent=2))
    return 0 if result['decision'].startswith('RESOURCE_SCREEN_SURVIVOR') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().out))
