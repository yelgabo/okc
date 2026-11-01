"""CPU-only two selected G-short paired-RNE compiler pairs; no GPU execution."""
import argparse, collections, hashlib, json, re, subprocess, types
from pathlib import Path
from prod_size_compile_g import bindings, sections


def run(root, candidate, out):
    raw = (root / 'muse_decode_layer/cand/g1b4_stage_release.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
    treatment = candidate.read_bytes()
    assert hashlib.sha256(treatment).hexdigest() == '19cfb4eeeb0dde21b142a23a880e8f921dfd02a63806a05649c40e6cc54e6a16'
    envs = [bindings(b.decode()) for b in (raw, treatment)]
    driver = (root / 'tools/local_cuda128_compile.py').read_text(encoding='utf8')
    old = "'-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1',\n               '-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_PRMTSC=1', '-DOKC_SC6W=' + str(args.sc6), *args.define"
    assert driver.count(old) == 1
    driver = driver.replace(old, '*args.define')
    old = "expressions = [b'&okc::moe_fused<false>', b'&okc::moe_fused<true>']"
    assert driver.count(old) == 1
    driver = driver.replace(old, "expressions = [b'&okc::moe_fused<false>']")
    ns = {'__name__': '_selected_short_counter_compile'}
    exec(compile(driver, '<existing-driver-selected-short-flags>', 'exec'), ns)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for key in ('tiny', 'tiny2'):
        defines = list(envs[0]['MOE_TINY_DEFINES' if key == 'tiny' else 'MOE_TINY2_DEFINES'])
        defines = [x for x in defines if not x.startswith('@')]
        if '-DOKC_FINPF=1' not in defines:
            defines.append('-DOKC_FINPF=1')
        common = ['-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_FNST=4', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1']
        arms = []
        for i, (arm, env) in enumerate(zip(('control', 'pair'), envs)):
            folder = out / (key + '-' + arm); folder.mkdir(parents=True, exist_ok=True)
            path = folder / 'input.cu'; path.write_bytes(env['_MA_ALT_SRCS']['moe_helper0'].encode())
            flags = common + defines + (['-DOKC_SHORT_PAIR_RNE=1'] if i else [])
            args = types.SimpleNamespace(source=path, out=folder, cache=Path('/root/.cache/okc-cuda128'), sc6=1,
                                         define=flags, forward_declare_wgs=True)
            ns['compile_source'](args)
            sass = subprocess.run([str(args.cache / 'cuda_nvdisasm-linux-x86_64-12.8.90-archive/bin/nvdisasm'),
                                   str(folder / 'device.cubin')], capture_output=True, text=True, check=True).stdout
            (folder / 'device.sass').write_text(sass)
            manifest = json.loads((folder / 'manifest.json').read_text())
            assert len(manifest['lowered']) == 1
            name = manifest['lowered'][0]
            ops = sections(folder / 'device.sass')[name]
            ptx = (folder / 'device.ptx').read_text()
            census = collections.Counter(re.sub(r'^@!?U?P\d+\s+', '', x).split()[0] for x in ops)
            arms.append(dict(manifest=manifest, instructions=len(ops), ops=ops, opcode_census=dict(census),
                             atom_fence_ptx=[x.strip() for x in ptx.splitlines() if re.search(r'\b(?:atom\.|red\.|fence\.|membar\.)', x)]))
        a, b = arms
        critical = lambda ops: [x for x in ops if re.search(r'\b(?:HGMMA|HMMA|HMUL2|FENCE|MEMBAR|BAR|SYNCS|WARPGROUP)[. ]', x)]
        records.append(dict(build=key, arms=arms, resources_equal=a['manifest']['resources'] == b['manifest']['resources'],
                            sass_identical=a['ops'] == b['ops'], math_fence_barrier_operands_equal=critical(a['ops']) == critical(b['ops']),
                            opcode_delta={x:b['opcode_census'].get(x,0)-a['opcode_census'].get(x,0) for x in sorted(set(a['opcode_census'])|set(b['opcode_census'])) if b['opcode_census'].get(x,0)!=a['opcode_census'].get(x,0)}))
    result = dict(control_sha256=hashlib.sha256(raw).hexdigest(), treatment_sha256=hashlib.sha256(treatment).hexdigest(),
                  records=records, gpu_execution=False, scope='Tiny/Tiny2 actual PDL-false entries only; device-only adapters exclude host launch/link/runtime')
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SHORT_PAIR_RNE_RESULT', json.dumps([{k:v for k,v in r.items() if k!='arms'} for r in records]))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('candidate',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.root,a.candidate,a.out)
