"""Frozen G224 prospective incumbent-relative offline resource screen."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import types

from g_expert_g224_prefix import BASE_SHA, NATIVE_SHA, sha
from test_g_expert_g224_prefix import ROOT, BASE, capture

FROZEN = ROOT / 'docs/steering/evidence/2026-11-02-worker-g224-chunk-phased'
CACHE = Path('/root/.cache/okc-cuda130-88')


def selections():
    assert sha(BASE) == BASE_SHA
    calls, env = capture(BASE, False)
    records = []
    originals = {}
    for tokens in (1, 8, 16, 32, 128, 256, 384):
        calls.clear()
        env['_MA_EXTS'] = {}
        state = dict(ma_sfx=('-DOKC_PRMTSC=1',), ma_sc6=True, ma_ext={})
        env['_fork_native_moe'](tokens, state)
        assert len(calls) == 1, (tokens, len(calls))
        call = calls[0]
        family = ('B4' if '-DOKC_B4FIN=1' in call['extra_cuda_cflags'] else 'MAIN') if '-DOKC_WGS=1' in call['extra_cuda_cflags'] else 'SHORT'
        record = dict(tokens=tokens, family=family, source_sha256=sha(call['cuda_sources']), cpp_sha256=sha(call['cpp_sources']), flags=call['extra_cuda_cflags'])
        records.append(record)
        if family != 'SHORT':
            assert record['source_sha256'] == NATIVE_SHA
            if family in originals:
                assert originals[family]['extra_cuda_cflags'] == call['extra_cuda_cflags']
                assert originals[family]['cuda_sources'] == call['cuda_sources']
            originals[family] = call
    assert {record['tokens']: record['family'] for record in records} == {1:'SHORT',8:'SHORT',16:'SHORT',32:'B4',128:'MAIN',256:'MAIN',384:'MAIN'}
    return records, originals


def gate(reference, frozen):
    fields = ('stack_bytes', 'spill_store_bytes', 'spill_load_bytes', 'barriers')
    assert set(reference['resources']) == set(frozen['resources'])
    results = {}
    for symbol, treatment in frozen['resources'].items():
        control = reference['resources'][symbol]
        rounded_blocks = ((16384 // (((treatment['registers'] * 32 + 255) // 256) * 256)) * 4) // 10
        results[symbol] = dict(no_worse={field: treatment[field] <= control[field] for field in fields}, registers_le96=treatment['registers'] <= 96, conditional_rounded_register_blocks=rounded_blocks)
        results[symbol]['pass'] = all(results[symbol]['no_worse'].values()) and results[symbol]['registers_le96'] and rounded_blocks >= 2
    return results


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    records, original = selections()
    summary = dict(scope='Original selected ba9 controls only; same fixed secondary NVRTC13/PTXAS13. No treatment compile, host NVCC, GPU, driver or target query.', started_epoch=time.time(), selections=records, records=[])
    driver_path = ROOT / 'docs/steering/evidence/2026-11-02-worker-g224-reference-readiness/original-control-nvrtc-driver.py'
    driver = driver_path.read_text()
    assert sha(driver) == 'e695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c'
    namespace = dict(__name__='_g224_original_resource_reference')
    exec(compile(driver, str(driver_path), 'exec'), namespace)
    summary['driver_sha256'] = sha(driver)
    summary['tools'] = {}
    for name, relative in (('ptxas','bin/ptxas'),('nvrtc','lib/libnvrtc.so.13'),('builtins','lib/libnvrtc-builtins.so.13.0')):
        path = CACHE / 'nvidia/cu13' / relative
        summary['tools'][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary['tools']['ptxas']['version'] = subprocess.run([str(CACHE/'nvidia/cu13/bin/ptxas'),'--version'], capture_output=True,text=True,check=True).stdout
    expected = dict(ptxas='daba837a68265cae38c832d13399b61dab811891de9b8914defddef143b849f2', nvrtc='a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11', builtins='91dcd944d01da9c0f08fff5d779db136a47f6d62bdb63bae900d2f481e92c3a2')
    assert all(summary['tools'][name]['sha256'] == value for name,value in expected.items())
    for family in ('B4','MAIN'):
        call = original[family]
        folder = out / family
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / 'original-selected.cu'
        source.write_bytes(call['cuda_sources'].encode())
        args = types.SimpleNamespace(source=source,out=folder,cache=CACHE,sc6=1,define=[flag for flag in call['extra_cuda_cflags'] if flag.startswith('-D')],forward_declare_wgs=True)
        namespace['compile_source'](args)
        reference = json.loads((folder/'manifest.json').read_text())
        frozen = json.loads((FROZEN/(family+'-manifest.json')).read_text())
        assert frozen['source_sha256'] == '53d09576f86e93b9baaa0145dcbdf1b60d8736743677d0a4de60e89864d18492'
        assert [option.replace('-DOKC_NW=16','-DOKC_NW=8') for option in reference['options']] == frozen['options']
        result = dict(family=family, reference=reference, actual_flags=call['extra_cuda_cflags'], frozen=frozen, gate=gate(reference,frozen))
        summary['records'].append(result)
        (out/'resource-result.json').write_text(json.dumps(summary,indent=2)+'\n')
    summary['decision'] = 'REFERENCE_RESOURCE_PASS_NOT_READY' if all(row['pass'] for record in summary['records'] for row in record['gate'].values()) else 'STOP_REFERENCE_RESOURCE_WORSENING'
    summary['finished_epoch'] = time.time()
    (out/'resource-result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(decision=summary['decision'], rows=[dict(family=record['family'], reference=record['reference']['resources'],gate=record['gate']) for record in summary['records']]),indent=2))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
