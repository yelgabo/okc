from pathlib import Path
import sys,json,types
sys.path.insert(0,'tools')
import wgs_shared_a_pipeline as g
from test_g_expert_g224_prefix import capture,BASE
raw='--raw' in sys.argv
out=Path('.build/shared-a-pipeline/compile-raw' if raw else '.build/shared-a-pipeline/compile');out.mkdir(parents=True,exist_ok=True)
s,n,pins=g.generate(out/'generated')
calls,e=capture(BASE,False);exec(s[len(BASE):],e);e['_MA_EXTS']={};e['_SSA_EXTS']={};calls.clear()
st=dict(ma_sfx=('-DOKC_PRMTSC=1',),ma_sc6=not raw,ma_ext={})
for t in (1,8,16,32,128,256,384):e['_fork_native_moe'](t,st)
driver=Path('docs/steering/evidence/2026-11-02-worker-g224-reference-readiness/original-control-nvrtc-driver.py').read_text()
assert g.sha(driver)=='e695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c'
driver=driver.replace('--std=c++17','--std=c++20')
witness='\n__device__ __constant__ unsigned ssa_layout[]={sizeof(Smem),((sizeof(Smem)+127)/128)*128,SSA_XOFF+WGS_NX*WGS_XS,JMAX,WGS_FIXF,NTL,SSA_AOFF,SSA_XOFF};\n'
driver=g.once(driver,'    declaration_adapter = None',"    device=device.replace('}  // namespace okc',"+repr(witness)+"+'}  // namespace okc')\n    declaration_adapter = None")
driver=g.once(driver,'    for name in expressions: check(lib.nvrtcAddNameExpression(program, name))','    for name in expressions+[b"&okc::ssa_layout"]: check(lib.nvrtcAddNameExpression(program, name))')
(out/'adapted-driver.py').write_text(driver)
env={'__name__':'ssa_fixed_secondary'};exec(driver,env)
r=dict(pins=pins,raw_scale_path=raw,scope='NVRTC13/PTXAS13 device-only C++20/header/declaration adapter; actual selected factory flags recorded; no host compilation/GPU',records=[])
for c in calls[3:]:
 family='B4' if '-DOKC_B4FIN=1' in c['extra_cuda_cflags'] else 'MAIN'
 folder=out/family;folder.mkdir(exist_ok=True);source=folder/'selected.cu';source.write_text(n)
 args=types.SimpleNamespace(source=source,out=folder,cache=Path('/root/.cache/okc-cuda130-88'),sc6=1,define=[f for f in c['extra_cuda_cflags'] if f.startswith('-D')],forward_declare_wgs=True)
 try:env['compile_source'](args)
 finally:
  rec=dict(family=family,flags=c['extra_cuda_cflags'],host_flags=c['extra_cflags'])
  if (folder/'manifest.json').exists():rec['manifest']=json.loads((folder/'manifest.json').read_text())
  r['records'].append(rec);(out/'compile-result.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(pins=pins,resources=[v['manifest'].get('resources') for v in r['records']]),indent=2))
