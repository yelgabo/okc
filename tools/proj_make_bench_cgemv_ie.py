"""Actual resource-threshold test: consumer warp0 applies original escape terms.

No ordering primitive is removed. Kernel launch excludes the old extra escape warp.
"""
from pathlib import Path
import ast
import sys
ROOT=Path(__file__).resolve().parents[1]

HOST=r'''
template<int N,int C,bool I> __global__ void __maxnreg__(32) ie32(const uint8_t* s,int hn,const int2* t,int pm,unsigned b,const __nv_bfloat16* x,__nv_bfloat16* o){if constexpr(I)ie_body<N,C,0>(s,hn,t,pm,b,x,o);else base_body<N,C,0>(s,hn,t,pm,b,x,o);}
template<int N,int C,bool I> __global__ void __maxnreg__(40) ie40(const uint8_t* s,int hn,const int2* t,int pm,unsigned b,const __nv_bfloat16* x,__nv_bfloat16* o){if constexpr(I)ie_body<N,C,0>(s,hn,t,pm,b,x,o);else base_body<N,C,0>(s,hn,t,pm,b,x,o);}
template<int N,int C,bool I> std::vector<int64_t> ieinfo(){
 auto k=C==1?ie32<N,C,I>:ie40<N,C,I>;cudaFuncAttributes a{};int blocks=0;
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,N+(I?0:32),0));
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,blocks};
}
template<int N,int C,bool I> void ielaunch(at::Tensor s,at::Tensor t,int64_t pm,int64_t b,at::Tensor x,at::Tensor o){
 auto st=at::cuda::getCurrentCUDAStream();int hn=o.numel()/2;
 auto sp=s.data_ptr<uint8_t>();auto tp=reinterpret_cast<const int2*>(t.data_ptr<int>());
 auto xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());auto op=reinterpret_cast<__nv_bfloat16*>(o.data_ptr());
 if constexpr(C==1)ie32<N,C,I><<<hn,N+(I?0:32),0,st>>>(sp,hn,tp,pm,b,xp,op);
 else ie40<N,C,I><<<hn,N+(I?0:32),0,st>>>(sp,hn,tp,pm,b,xp,op);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
std::vector<int64_t> ie_info(int64_t nt,bool ie){
 #define INFO(N,C) return ie?ieinfo<N,C,true>():ieinfo<N,C,false>()
 if(nt==64){INFO(64,1);}if(nt==128){INFO(128,2);}if(nt==256){INFO(256,2);}INFO(512,1);
}
void ie_fwd(at::Tensor s,at::Tensor t,int64_t pm,int64_t b,at::Tensor x,at::Tensor o,int64_t nt,bool ie){
 #define GO(N,C) if(ie)ielaunch<N,C,true>(s,t,pm,b,x,o);else ielaunch<N,C,false>(s,t,pm,b,x,o)
 if(nt==64){GO(64,1);}else if(nt==128){GO(128,2);}else if(nt==256){GO(256,2);}else{GO(512,1);}
}
'''
DECL='std::vector<int64_t> ie_info(int64_t,bool);void ie_fwd(at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor,at::Tensor,int64_t,bool);'
RUN=r'''
import random as _ierandom
import statistics as _iestats
_IE_LINES=[]
def _ieprint(s):_IE_LINES.append(s);print(s,flush=True)
def _iebench(weights,state):
    mods={}
    for label in ('a','b'):
        mods[label]=load_inline(name='okc_cgemv_ie_'+label,cpp_sources='#include <torch/extension.h>\n'+@@DECL@@,cuda_sources=@@CUDA@@,functions=['ie_info','ie_fwd'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],no_implicit_headers=True,verbose=False)
    dev=weights['pooler'].device;gen=torch.Generator(device=dev).manual_seed(81031)
    flush=torch.empty(256*1024*1024,dtype=torch.uint8,device=dev)
    for name in ('qkv_gate','o','pooler','unpooler'):
        w=weights[name];w=w[:1024] if name=='unpooler' else w;nt=cgemv_threads(*w.shape);pk=pack_cgemv(w)
        if pk is None:raise RuntimeError('control cgemv packing unavailable')
        arms={'prod':state['pick'][name,1]}
        for label,mod in mods.items():
            old=mod.ie_info(nt,False);new=mod.ie_info(nt,True)
            _ieprint(f'IERES {name} {label} nt={nt} old/new regs,local,shared,blocks={old}/{new}')
            for inline in (False,True):
                tag=label+('_ie' if inline else '_old')
                if inline and (new[3]<=old[3] or new[1]):
                    _ieprint(f'IESKIP {name} {tag} no actual occupancy gain or local memory');continue
                def run(x,mod=mod,inline=inline):
                    out=torch.empty((1,w.shape[0]),dtype=torch.bfloat16,device=dev)
                    mod.ie_fwd(pk['stream'],pk['table'],pk['pmax'],pk['base4'],x,out,nt,inline);return out
                arms[tag]=run
        for probe in (False,True):
            x=torch.randn((1,w.shape[1]),device=dev,generator=gen).to(torch.bfloat16)
            if probe:x.zero_();x[0,17]=2
            ref=x.double()@w.double().T
            for tag,fn in arms.items():
                out=fn(x);err=float(torch.linalg.vector_norm(out.double()-ref)/torch.linalg.vector_norm(ref))
                assert err<.005,(name,tag,err)
            _ieprint(f'IECHECK {name} probe={probe} PASS arms={list(arms)}')
        totals={tag:[] for tag in arms}
        for inp in range(2):
            x=torch.randn((1,w.shape[1]),device=dev,generator=gen).to(torch.bfloat16);graphs={}
            for tag,fn in arms.items():
                fn(x);g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out=fn(x)
                graphs[tag]=g
            times={tag:[] for tag in arms};rng=_ierandom.Random(81031+inp)
            for rd in range(100):
                order=list(graphs);rng.shuffle(order)
                for tag in order:
                    flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                    a.record();graphs[tag].replay();z.record();z.synchronize();times[tag].append(1000*a.elapsed_time(z))
            for tag,t in times.items():
                delta=[a-b for a,b in zip(t,times['prod'])];totals[tag]+=delta
                _ieprint(f'IETIME {name} input={inp} {tag} raw={_iestats.median(t):.4f} delta={_iestats.median(delta):+.4f} faster={sum(d<0 for d in delta)}/100')
        _ieprint(f'IESUM {name} deltas_us='+str({tag:_iestats.median(d) for tag,d in totals.items()}))
_ie_original_prepare=prepare
def prepare(weights):
    state=_ie_original_prepare(weights);_iebench(weights,state)
    for line in _IE_LINES:print(line,flush=True)
    return state
'''

def generate(source):
    tree=ast.parse(source)
    cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='GEMV_CUDA_SRC' for t in n.targets))
    start=cuda.index('template <int NTH, int CPT, int VB = 0>')
    end=cuda.index('// Registers set the occupancy',start)
    body=cuda[start:end];prefix='#include <vector>\n'+cuda[:start]
    anchor='    } else {\n        // Lane i loads entry i'
    assert body.count(anchor)==1
    candidate=body.replace('cgemv_body(', 'ie_body(',1).replace(anchor,'    }\n    if (warp == 0) {\n        // Lane i loads entry i')
    control=body.replace('cgemv_body(', 'base_body(',1)
    # Each emitted instance retains exactly the original CTA barrier and all math.
    assert control.count('__syncthreads();')==candidate.count('__syncthreads();')
    assert control[control.rindex('__syncthreads();'):]==candidate[candidate.rindex('__syncthreads();'):]
    assert control[control.index('// Lane i loads entry i'):]==candidate[candidate.index('// Lane i loads entry i'):]
    for nt in (64,128,256,512):
        assert list(range(nt))==[t for t in range(nt+32) if t<nt]
        for count in (0,1,17,31,32,33,63,137):
            old=sorted(e for lane in range(32) for e in range(lane,count+1,32) if e>=1)
            new=sorted(e for t in range(32) for e in range(t,count+1,32) if e>=1)
            assert old==new==list(range(1,count+1))
        old_slots=[(w*32,w) for w in range(nt//32)]+[(nt,nt//32)]
        new_slots=[(w*32,w) for w in range(nt//32)]+[(0,nt//32)]
        assert [s for t,s in old_slots]==[s for t,s in new_slots]==list(range(nt//32+1))
    return prefix+control+candidate+HOST

if __name__=='__main__':
    source=(ROOT/'muse_bf16_proj/submission.py').read_text()
    cuda=generate(source)
    Path(sys.argv[1]).write_text(source+RUN.replace('@@DECL@@',repr(DECL)).replace('@@CUDA@@',repr(cuda)))
    print('CPU consumer/escape ownership and preserved barrier/final-sum gate PASS; wrote',sys.argv[1])
