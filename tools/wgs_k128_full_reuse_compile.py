"""Actual selected source/flags, fixed secondary CUDA13 compile; no GPU."""
import argparse
import json
from pathlib import Path
import types
import wgs_k128_full_reuse as gen
from test_wgs_k128_full_reuse import selected
from test_g_expert_g224_prefix import capture, BASE

ROOT=Path(__file__).resolve().parents[1]

def run(out,reference=False):
    out.mkdir(parents=True,exist_ok=True)
    candidate,native,pins=gen.generate(out/'generated')
    calls,_,actual=selected();assert actual==native
    if reference:
        calls,_=capture(BASE,False)
        native=gen.literal(BASE,'MOE_CUDA_SRC')
    driver=(ROOT/'docs/steering/evidence/2026-11-02-worker-g224-reference-readiness/original-control-nvrtc-driver.py').read_text()
    assert gen.sha(driver)=='e695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c'
    driver=driver.replace('--std=c++17','--std=c++20')
    anchor='    declaration_adapter = None'
    assert driver.count(anchor)==1
    witness='\n__device__ __constant__ unsigned k128_layout[] = {sizeof(Smem), ((sizeof(Smem)+127)/128)*128, WGS_NS*K128_RING_BYTES+WGS_NX*WGS_XS, JMAX, WGS_FIXF};\n'
    if reference:witness=witness.replace('K128_RING_BYTES','UNIT_BYTES')
    driver=driver.replace(anchor,"    device = device.replace('}  // namespace okc', "+repr(witness)+" + '}  // namespace okc')\n"+anchor)
    anchor='    for name in expressions: check(lib.nvrtcAddNameExpression(program, name))'
    assert driver.count(anchor)==1
    driver=driver.replace(anchor,'    for name in expressions + [b"&okc::k128_layout"]: check(lib.nvrtcAddNameExpression(program, name))')
    ns={'__name__':'fixed_k128_secondary'};exec(compile(driver,'<fixed-k128-secondary>','exec'),ns)
    (out/'adapted-driver.py').write_text(driver)
    result=dict(pins=pins,reference=reference,selected_native_sha256=gen.sha(native),scope='Fixed installed NVRTC13/PTXAS13 device-only with C++20/declaration adapter; actual factory flags recorded; host compilation/target occupancy/numerics unmeasured. pins describes generated treatment, selected_native_sha256 describes this compile.',records=[])
    for call in calls[3:]:
        family='B4' if '-DOKC_B4FIN=1' in call['extra_cuda_cflags'] else 'MAIN'
        folder=out/family;folder.mkdir(parents=True,exist_ok=True)
        source=folder/'selected.cu';source.write_text(native)
        args=types.SimpleNamespace(source=source,out=folder,cache=Path('/root/.cache/okc-cuda130-88'),sc6=1,define=[f for f in call['extra_cuda_cflags'] if f.startswith('-D')],forward_declare_wgs=True)
        try:ns['compile_source'](args)
        finally:
            record=dict(family=family,flags=call['extra_cuda_cflags'],host_flags=call.get('extra_cflags',[]),reference=reference)
            if (folder/'manifest.json').exists():record['manifest']=json.loads((folder/'manifest.json').read_text())
            result['records'].append(record)
            (out/'compile-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pins=pins,resources=[r['manifest']['resources'] for r in result['records']]),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',action='store_true');a=p.parse_args();run(a.out,a.reference)
