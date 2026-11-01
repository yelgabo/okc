"""Disclosed device-only VP2 NVRTC gate using the existing owned compile driver."""
from pathlib import Path
import argparse
import hashlib
import ast

def extract(source):
    marker='void vp2_launch('
    assert source.count(marker)==1
    device=source[:source.index(marker)]+'\n} // b1vp2\n} // vfgns\n'
    anchor='void vfg_launch('
    assert device.count(anchor)==1
    a=device.index(anchor);z=device.index('{',a)+1;depth=1
    while depth:depth+=(device[z]=='{')-(device[z]=='}');z+=1
    device=device[:a]+device[z:]
    for header in ('stdexcept','cuda.h','cudaTypedefs.h'):
        device=device.replace('#include <'+header+'>\n','')
    return device

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--cache',type=Path,default=Path('/root/.cache/okc-cuda128'));a=p.parse_args()
    driver=Path(__file__).with_name('moe_decoder_compile.py').read_text(encoding='utf8')
    driver=driver.replace("expressions=['&moe_main_kernel<47>','&moe_main_kernel<51>','&sk_8_3_0::moe_fused_kernel<1>']","expressions=['&vfgns::b1vp2::vp2_kernel']")
    driver=driver.replace("options=['--gpu-architecture=compute_90a'","options=['-DZH_SCH=0','--gpu-architecture=compute_90a'")
    driver=driver.replace("adapters=['remove torch/ATen includes and host extension/launch bodies','cstdint virtual fixed-width declarations','select existing sk8/3 only']","adapters=['drop VFG/VP2 host launch wrappers and stdexcept; retain device prefix','fixed-width virtual stdint','select actual vfgns::b1vp2::vp2_kernel; SCH0']")
    h=next(n for n in ast.walk(ast.parse(driver)) if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='hdr' for t in n.targets))
    driver=driver.replace(ast.get_source_segment(driver,h),'hdr='+repr('#pragma once\n#include <cuda/std/cstdint>\n'))
    ns={'__name__':'_vp2_device_gate'};exec(compile(driver,'<owned-compiler-adapter>','exec'),ns)
    ns['extract']=extract
    ns['run'](a)
