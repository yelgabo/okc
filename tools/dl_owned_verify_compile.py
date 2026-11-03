"""Bounded device-only screen of selected kernels affected by whole-family pruning."""
import argparse,ast
from pathlib import Path
from dl_owned_verify_production import literal,function

def extract(source,kind):
    if kind=='route':
        s=literal(source,'_CUDA_SRC')
        a=s.index('std::vector<torch::Tensor> route_align(') if 'std::vector<torch::Tensor> route_align(' in s else s.index('std::vector<torch::Tensor> route_fwd(')
        s=s[:a]
        if s.count('namespace {')>s.count('} // namespace')+s.count('}  // namespace'):s+='\n} // namespace\n'
        for anchor in ('void launch_ex(','void launch_ex_pdl(','void router_launch(','void proj_launch(','void proj_small(','void router_mma('):
            if anchor in s:
                body=function(s,anchor);s=s.replace(body,body[:body.index('{')]+';')
        s='#define INFINITY (__int_as_float(0x7f800000))\nnamespace std {template<class A,class B> constexpr bool is_same_v=false;template<class A> constexpr bool is_same_v<A,A>=true;}\n'+s
        s=s.replace('is_same_v<A,A>=true','is_same_v<A,A> = true')
    elif kind=='boundary':
        s=literal(source,'_BND_SRC');s=s[:s.index('template <typename Kern, typename... Args>\nvoid launch_cl(')]+'\n} // bnd\n} // namespace\n'
    else:
        s='struct alignas(64) CUtensorMap { unsigned long long opaque[16]; };\n#define SK_DEVICE_ONLY 1\n'+literal(source,'_SKP_SRC')+'\n'+literal(source,'_SKP_WD_SRC')
    for h in ('torch/extension.h','ATen/cuda/CUDAContext.h','cuda.h','cudaTypedefs.h'):
        s=s.replace('#include <'+h+'>','#include <cuda_runtime.h>' if h=='torch/extension.h' else '')
    return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('kind',choices=('route','boundary','wd'))
    p.add_argument('--cache',type=Path,default=Path('/root/.cache/okc-cuda128'));args=p.parse_args()
    driver=Path(__file__).with_name('moe_decoder_compile.py').read_text(encoding='utf8')
    expressions={'route':'&route_kernel<float,true>','boundary':'&bnd::attn_boundary_geo_kernel<1,512>','wd':'&wd_kernel<1,128,3,1,false,8,__nv_bfloat16,0,true,0>'}
    driver=driver.replace("expressions=['&moe_main_kernel<47>','&moe_main_kernel<51>','&sk_8_3_0::moe_fused_kernel<1>']","expressions=["+repr(expressions[args.kind])+"]")
    h=next(n for n in ast.walk(ast.parse(driver)) if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='hdr' for t in n.targets))
    driver=driver.replace(ast.get_source_segment(driver,h),'hdr='+repr('#pragma once\n#include <cuda/std/cstdint>\n'))
    driver=driver.replace("options=['--gpu-architecture=compute_90a'","options=['-DSK_PDL=1','--gpu-architecture=compute_90a'")
    driver=driver.replace("adapters=['remove torch/ATen includes and host extension/launch bodies','cstdint virtual fixed-width declarations','select existing sk8/3 only']","adapters=['device-only family extraction; Torch/ATen/driver header adapters','CUDA cstdint; route is_same_v alias','selected original device entry; no host API proof']")
    ns={'__name__':'_verify_family_device_gate'};exec(compile(driver,'<owned-device-compiler-adapter>','exec'),ns)
    ns['extract']=lambda s:extract(s,args.kind)
    ns['run'](args)
