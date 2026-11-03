"""Fixed two-build/four-entry secondary NVRTC13/PTXAS13 gate, no GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import types

from g_expert_g224_prefix import generate
from test_g_expert_g224_prefix import ROOT, BASE, capture, source_contract
from g_expert_g224_prefix_nvrtc_layout import adapt_driver, adapt_device, inspect_ptx, PROBES


def run(out, main_only=False):
    variant, native, generated = generate(ROOT, out / 'generated')
    source_contract(native)
    calls, _ = capture(variant, True)
    selected = [call for call in calls if '-DOKC_WGS=1' in call['extra_cuda_cflags']]
    assert len(selected) == 2
    if main_only:
        selected = [call for call in selected if '-DOKC_B4FIN=1' not in call['extra_cuda_cflags']]
        assert len(selected) == 1
    driver_path = ROOT / '.build/research-gamma13/artifacts/adapted-driver.py'
    driver = driver_path.read_text()
    namespace = {'__name__': '_fixed_g224_prefix_secondary13', 'adapt_device': adapt_device, 'inspect_ptx': inspect_ptx, 'PROBES': PROBES}
    adapted = adapt_driver(driver)
    exec(compile(adapted, str(driver_path), 'exec'), namespace)
    result = dict(scope='CPU-only NVRTC13/PTXAS13 compute90a/sm90a device prefix; virtual type_traits/stdint and declaration-only wgs_cta_of adapter. No host compile/link, NVCC, driver API, GPU, timing or target occupancy.',
                  generator=generated, driver_sha256=hashlib.sha256(driver.encode()).hexdigest(), adapted_driver_sha256=hashlib.sha256(adapted.encode()).hexdigest(), layout_adapter='Captain-authorized two RTC-only offset assertions replaced by five emitted same-Smem probes; actual CUDA/NVCC source unchanged', started_epoch=time.time(), records=[])
    summary = out / 'compile-result.json'
    cache = Path('/root/.cache/okc-cuda130-88')
    for call in selected:
        family = 'B4' if '-DOKC_B4FIN=1' in call['extra_cuda_cflags'] else 'MAIN'
        folder = out / family
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / 'selected.cu'
        source.write_bytes(native.encode())
        record = dict(family=family, actual_flags=call['extra_cuda_cflags'], source_sha256=hashlib.sha256(native.encode()).hexdigest())
        result['records'].append(record)
        args = types.SimpleNamespace(source=source, out=folder, cache=cache, sc6=1,
                    define=[flag for flag in call['extra_cuda_cflags'] if flag.startswith('-D')], forward_declare_wgs=True)
        try:
            namespace['compile_source'](args)
            manifest = json.loads((folder / 'manifest.json').read_text())
            record['manifest'] = manifest
            log = (folder / 'ptxas.log').read_text()
            record['ptxas_log'] = log
            record['resources_pass'] = all(value['stack_bytes']==0 and value['spill_store_bytes']==0 and value['spill_load_bytes']==0 for value in manifest['resources'].values())
            assert record['resources_pass'], 'G224_STOP emitted stack/spills'
            record['rounded_register_blocks'] = {symbol: ((16384 // (((value['registers'] * 32 + 255) // 256) * 256)) * 4) // 10 for symbol,value in manifest['resources'].items()}
            assert all(blocks >= 2 for blocks in record['rounded_register_blocks'].values()), 'G224_STOP rounded SM90 register admission'
            ptx = (folder / 'device.ptx').read_text()
            assert '.maxntid 320' in ptx and '.minnctapersm 2' in ptx
            disasm = subprocess.run([str(cache/'cuda_nvdisasm-linux-x86_64-13.0.85-archive/bin/nvdisasm'), str(folder/'device.cubin')], capture_output=True, text=True, check=True)
            (folder/'device.sass').write_text(disasm.stdout)
            record['artifact_sha256'] = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('selected.cu','device.ptx','device.cubin','device.sass','ptxas.log','nvrtc.log')}
        except Exception as error:
            record['stop'] = str(error)
            if (folder/'manifest.json').exists(): record['manifest'] = json.loads((folder/'manifest.json').read_text())
            result['decision'] = 'STOP'
            result['finished_epoch'] = time.time()
            summary.write_text(json.dumps(result,indent=2)+'\n')
            raise
        summary.write_text(json.dumps(result,indent=2)+'\n')
    result['decision'] = 'OFFLINE_RESOURCE_PASS_NOT_TARGET_QUALIFIED'
    result['finished_epoch'] = time.time()
    summary.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--main-only', action='store_true', help='Complete the fixed MAIN pair after separately archived B4 STOP')
    args = parser.parse_args()
    run(args.out, args.main_only)
