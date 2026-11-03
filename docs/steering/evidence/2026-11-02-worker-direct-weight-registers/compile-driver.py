from pathlib import Path
import sys,json,types
sys.path.insert(0,'tools')
import wgs_direct_weight_registers as g
from test_wgs_direct_weight_registers import selected
raw='--raw' in sys.argv
out=Path('.build/direct-weight-registers/compile-raw' if raw else '.build/direct-weight-registers/compile');out.mkdir(parents=True,exist_ok=True)
s,n,pins=g.generate(out/'generated');calls,_,actual=selected(not raw);assert actual==n
driver=Path('docs/steering/evidence/2026-11-02-worker-g224-reference-readiness/original-control-nvrtc-driver.py').read_text()
assert g.sha(driver)=='e695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c'
driver=driver.replace('--std=c++17','--std=c++20')
witness='\n__device__ __constant__ unsigned dw_layout[]={sizeof(Smem),((sizeof(Smem)+127)/128)*128,WGS_NX*WGS_XS,JMAX,WGS_FIXF,NTL,WGS_NX,WGS_XS};\n'
driver=g.once(driver,'    declaration_adapter = None',"    device=device.replace('}  // namespace okc',"+repr(witness)+"+'}  // namespace okc')\n    declaration_adapter = None")
driver=g.once(driver,'    for name in expressions: check(lib.nvrtcAddNameExpression(program, name))','    for name in expressions+[b"&okc::dw_layout"]: check(lib.nvrtcAddNameExpression(program, name))')
(out/'adapted-driver.py').write_text(driver)
env={'__name__':'dw_fixed_secondary'};exec(driver,env)
result=dict(pins=pins,raw_scale_path=raw,scope='NVRTC13/PTXAS13 secondary device-only C++20 with virtual-header/declaration adapter; actual factory flags recorded; no host compilation or GPU',records=[])
for c in calls[3:]:
    family='B4' if '-DOKC_B4FIN=1' in c['extra_cuda_cflags'] else 'MAIN'
    dest=out/family;dest.mkdir(exist_ok=True);source=dest/'selected.cu';source.write_text(n)
    args=types.SimpleNamespace(source=source,out=dest,cache=Path('/root/.cache/okc-cuda130-88'),sc6=int(not raw),define=[f for f in c['extra_cuda_cflags'] if f.startswith('-D')],forward_declare_wgs=True)
    try:env['compile_source'](args)
    finally:
        rec=dict(family=family,flags=c['extra_cuda_cflags'],host_flags=c['extra_cflags'])
        if (dest/'manifest.json').exists():rec['manifest']=json.loads((dest/'manifest.json').read_text())
        result['records'].append(rec);(out/'compile-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(pins=pins,resources=[r['manifest'].get('resources') for r in result['records']]),indent=2))
