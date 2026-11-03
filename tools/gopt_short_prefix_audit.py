"""Check whether the proposed serial WGS prefix survives selected helper flags."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess

from d32_b4_prefix_audit import audit, body, HELPER_SHA
from g_expert_g224_prefix import sha
from prod_size_compile_g import bindings

ROOT = Path(__file__).resolve().parents[1]


def run(out):
    source = (ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py').read_text()
    factory = audit(source)
    helper = bindings(source)['_MA_ALT_SRCS']['moe_helper0']
    assert sha(helper) == HELPER_SHA
    device = helper.split('#include <torch/extension.h>')[0]
    work = ROOT / '.build/gopt-short-prefix'
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (work / 'helper.cu').write_text(device)
    records = []
    for index, build in enumerate(r for r in factory['builds'] if r['native_sha256'] == HELPER_SHA):
        for counterfactual in (False, True) if index == 0 else (False,):
            name = ('tiny', 'tiny2', 'mid')[index] + ('-forced-wgs' if counterfactual else '')
            flags = build['flags'] + (['-DOKC_WGS=1'] if counterfactual else [])
            command = ['docker', 'run', '--rm', '--entrypoint', '', '--platform', 'linux/arm64',
                       '-v', str(work) + ':/w', 'nvidia/cuda:12.8.1-devel-ubuntu24.04',
                       'nvcc', *flags, '-E', '/w/helper.cu', '-o', '/w/' + name + '.i']
            process = subprocess.run(command, capture_output=True, text=True)
            (out / (name + '.log')).write_text(process.stdout + process.stderr)
            process.check_returncode()
            preprocessed = (work / (name + '.i')).read_text()
            present = 'void wgs_ranges(' in preprocessed
            assert present is counterfactual
            assert ('void wgs_xproducer(' in preprocessed) is counterfactual
            assert ('void wgs_prefix(' in preprocessed) is counterfactual
            kernel = body(preprocessed, 'moe_fused(Params p)')
            assert 'ws_helper(p, S,' in kernel
            assert 'S.wsr[0] = muldiv_g(nA, c, G, rG);' in kernel
            assert 'S.wsr[3] = muldiv_g(nB, c + 1, G, rG);' in kernel
            if not counterfactual:
                assert 'wgs_ranges(' not in preprocessed and 'wgs_prefix(' not in preprocessed
                assert 'wgs_xproducer(' not in kernel
            (out / (name + '.i.gz')).write_bytes(gzip.compress(preprocessed.encode(), mtime=0))
            records.append(dict(name=name, tokens=build['tokens'], module=build['module'],
                                flags=flags, counterfactual=counterfactual,
                                native_sha256=sha(helper), device_prefix_sha256=sha(device),
                                preprocessed_sha256=sha(preprocessed), exit_code=process.returncode,
                                wgs_functions_present=present, ws_kernel_body=kernel))
            print(name, 'WGS functions present:', present, flush=True)
    result = dict(base_sha256=factory['base_sha256'], helper_sha256=sha(helper), records=records,
                  actual_factory_builds=factory['builds'],
                  decision='Close GOPT-SHORT-02: proposed WGS prefix is preprocessed out of all selected helper builds.',
                  scope='Exact selected flags; device prefix preprocessing only. Host extension excluded. Synthetic WGS override is a negative control, not a runnable candidate.',
                  candidate_created=False, gpu_execution=False)
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
