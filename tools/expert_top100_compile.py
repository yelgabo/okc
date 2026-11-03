"""Fixed B4/MAIN device-only secondary compiler screen; no GPU or installs."""
import argparse
import json
from pathlib import Path
import types

from expert_top100_pregamma import ROOT, generate, sha


def run(out):
    generated = out / 'generated'
    m = generate(generated)
    driver = (ROOT / 'tools/local_cuda128_compile.py').read_text()
    old = "'-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1',\n               '-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_PRMTSC=1', '-DOKC_SC6W=' + str(args.sc6), *args.define"
    assert driver.count(old) == 1
    adapted = driver.replace(old, '*args.define')
    env = dict(__name__='_expert_top100_secondary')
    exec(compile(adapted, '<existing-driver-actual-selected-flags>', 'exec'), env)
    result = dict(gpu=False, driver_sha256=sha(driver), adapted_driver_sha256=sha(adapted), records=[],
                  scope='NVRTC/PTXAS12.8 device prefix, established virtual headers and declaration adapter; no host/link/runtime qualification')
    for selected in m['selection']:
        if not selected['changed']:
            continue
        family = 'b4' if '-DOKC_B4FIN=1' in selected['flags'] else 'main'
        row = dict(family=family, arms={})
        result['records'].append(row)
        for arm in ('control', 'candidate'):
            folder = out / (family + '-' + arm)
            args = types.SimpleNamespace(source=generated / (arm + '.cu'), out=folder,
                cache=Path('/root/.cache/okc-cuda128'), sc6=1,
                define=[f for f in selected['flags'] if f.startswith('-D')], forward_declare_wgs=True)
            try:
                env['compile_source'](args)
                manifest = json.loads((folder / 'manifest.json').read_text())
                row['arms'][arm] = manifest
            except Exception as error:
                result.update(decision='STOP_COMPILER', error=repr(error))
                (out / 'compile.json').write_text(json.dumps(result, indent=2) + '\n')
                raise
        a, b = row['arms']['control'], row['arms']['candidate']
        assert set(a['resources']) == set(b['resources']) and len(a['resources']) == 2
        row['resource_gate'] = all(b['resources'][n][k] <= a['resources'][n][k]
            for n in a['resources'] for k in ('registers', 'stack_bytes', 'spill_store_bytes', 'spill_load_bytes'))
        row['same_bounds_barriers'] = a['ptx_launch_bounds'] == b['ptx_launch_bounds'] and all(a['resources'][n]['barriers'] == b['resources'][n]['barriers'] for n in a['resources'])
        row['no_new_serialization'] = b['ptxas_wgmma_serialization_warnings'] <= a['ptxas_wgmma_serialization_warnings']
        (out / 'compile.json').write_text(json.dumps(result, indent=2) + '\n')
    result['decision'] = 'SOURCE_COMPILER_SURVIVOR_UNQUALIFIED' if all(r['resource_gate'] and r['same_bounds_barriers'] and r['no_new_serialization'] for r in result['records']) else 'STOP_RESOURCE'
    (out / 'compile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['decision'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
