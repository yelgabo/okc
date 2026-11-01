"""CPU-only sm90a device equivalence gate for owned hybrid native kernels."""
import argparse,ast,json,re,hashlib
from pathlib import Path

def literal(s,name):
 return next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)

def extract(s,kind):
 v=literal(s,'_ONE_SRC' if kind=='one' else '_MOE1_SRC')
 marker='template <int L, int MODE>\nstruct Instance {' if kind=='one' else 'template <int MODE>\nvoid launch_main('
 assert v.count(marker)==1;v=v[:v.index(marker)]
 if kind=='moe':
  a=v.index('template <typename Kern>\nvoid moe1_launch(');b=v.index('__device__ __forceinline__ void prefetch_l2_hint',a);v=v[:a]+v[b:]
 v=re.sub(r'#include <(?:torch|ATen)/[^>]*>\n','',v)
 v=re.sub(r'#include <(?:algorithm|vector)>\n','',v)
 v=v.replace('#include <cstdint>','#include <stdint.h>')
 if kind=='one':
  old='constexpr float NEG_INF = -INFINITY;';assert v.count(old)==1;v=v.replace(old,'#define NEG_INF (__int_as_float((int)0xff800000u))')
 return '#include <cuda_runtime.h>\n'+v+'\n} // native inner namespace\n} // anonymous namespace\n'

def run(a):
 driver=(a.root/'tools/moe_decoder_compile.py').read_text(encoding='utf8')
 old="expressions=['&moe_main_kernel<47>','&moe_main_kernel<51>','&sk_8_3_0::moe_fused_kernel<1>']"
 ex=['&one::attn_one_kernel<8,3>','&one::attn_one_kernel<8,51>'] if a.kind=='one' else ['&moe1::moe_main_kernel<51>']
 assert driver.count(old)==1;driver=driver.replace(old,'expressions='+repr(ex))
 if a.kind=='one':driver=driver.replace("options=['--gpu-architecture=compute_90a'","options=['-DONE_TRIG=1','--gpu-architecture=compute_90a'")
 driver=driver.replace("adapters=['remove torch/ATen includes and host extension/launch bodies','cstdint virtual fixed-width declarations','select existing sk8/3 only']","adapters=['owned selected literal; remove host launcher/Instance only; preserve all device bodies','Torch/ATen and host std includes removed; CUDA runtime and fixed-width declarations','ONE infinity declaration adapter and DONE_TRIG=1; no host API proof']")
 ns={'__name__':'owned_comment_compile'};exec(compile(driver,'<owned-native-compile-driver>','exec'),ns);ns['extract']=lambda s:extract(s,a.kind);ns['run'](a)
 manifest=json.loads((a.out/'manifest.json').read_text());manifest['driver_source_sha256']=hashlib.sha256((a.root/'tools/moe_decoder_compile.py').read_bytes()).hexdigest();manifest['kind']=a.kind;(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('kind',choices=['one','moe']);p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,default=Path('/root/.cache/okc-cuda128'));run(p.parse_args())
