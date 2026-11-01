"""Generate pinned existing C2 device instantiations; no changed kernel body."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    raw=a.source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4'
    spec=importlib.util.spec_from_file_location('_pinned_c2_source',a.source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    sknames={'qkv8F','qkv16F','qkv32','o16F','o32F','pool16F','pool32F','unp16F','unp32F','slice8F'}
    m.SK_CONFIGS=[c for c in m.SK_CONFIGS if c[0] in sknames]
    assert {c[0] for c in m.SK_CONFIGS}==sknames
    wdnames=['b1n128c3s8h','r64c8s8k','r128c8s6k']
    m.WD_NAMES=wdnames;m.WD_GEOM={k:m.WD_GEOM[k] for k in wdnames}
    # Device-only is the existing official exporter path, without GEMV's unrelated
    # CUDA-core explicit instantiations. Common helpers/device definitions stay.
    cu='#define SK_DEVICE_ONLY 1\n'+m.sk_source()+m.wd_source()+m.rt_source()+m.rd_source()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(cu,encoding='utf8',newline='\n')
    manifest=dict(source_sha256=hashlib.sha256(raw).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),
        sk_configs={c[0]:m._sk_targs(c) for c in m.SK_CONFIGS},wd_configs={k:m._wd_targs(k) for k in wdnames},
        rt_configs=m.RT_GEOM,rt_mt=m.RT_MT,rd_configs=m.RD_GEOM,
        limitations='No kernel-body changes. SK_DEVICE_ONLY excludes host wrappers. RT emits9 existing MT/config combinations; only3 are active. No GEMV/Lt/CUTLASS device instantiation.',gpu_execution=False)
    a.output.with_suffix('.export.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
