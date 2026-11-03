"""OFF-by-default stage-local sparse E7 records, after measured global-SX loss.

Capacity is the maximum actual stage/CTA record count, never a tuned cap.
One weight bulk transfer includes payload, headers and padded record arena.
"""
import argparse,ast,hashlib,json
from pathlib import Path
import proj_sparse_xor as previous

STAGE_BIT=8192
NAMES=previous.NAMES
PIN=previous.PIN

STAGE_HELPER=r'''
__device__ __forceinline__ unsigned sxs_lds16(uint32_t address) {
    unsigned value;
    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return value;
}
// Same warp-uniform record order/masks as the global variant. Every record
// access is now part of the credited ring stage, and precedes its empty arrive.
__device__ __forceinline__ void sxs_repair(uint32_t (&b)[8],uint32_t h,
                                          uint32_t arena,int lane) {
    const unsigned count=h&1023u,start=h>>10;
    for(unsigned i=0;i<count;++i){
        const unsigned rec=sxs_lds16(arena+2*(start+i));
        const unsigned reg=(rec>>5)&7u;
        const uint32_t x=((rec>>9)&127u)<<(8+16*((rec>>8)&1u));
        const uint32_t mask=(rec&31u)==(unsigned)lane?x:0u;
#pragma unroll
        for(int r=0;r<8;++r)b[r]^=reg==(unsigned)r?mask:0u;
    }
}
'''


def patch_cuda(original):
    s=previous.patch_cuda(original);once=previous.once
    s=once(s,'// Eight 12-bit weights ->',STAGE_HELPER+'\n// Eight 12-bit weights ->')
    s=once(s,'    static constexpr bool SX=(NACC & 4096)!=0;',
        '    static constexpr bool SX=(NACC & 4096)!=0;\n    static constexpr bool STAGE=(NACC & 8192)!=0;')
    s=once(s,'    static_assert(!SX ||',
        '    static_assert(!STAGE || SX,"stage records require sparse XOR");\n    static_assert(!SX ||')
    s=once(s,'    constexpr int NW = C::NW, NC = C::NC, R = C::R, UPS = C::UPS, TU = C::TU, WBY = C::WBY, XBY = C::XBY;',
        '    constexpr int NW = C::NW, NC = C::NC, R = C::R, UPS = C::UPS, TU = C::TU, XBY = C::XBY;\n'
        '    const int WBY=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);')
    s=once(s,'    constexpr int SST = C::SST, TOKS = C::TOKS, RP = C::RP, CP = C::CP, NESC = C::NESC, CSZ = C::CSZ;',
        '    const int SST=C::STAGE?((XBY+WBY+1023)&~1023):C::SST;\n'
        '    constexpr int TOKS = C::TOKS, RP = C::RP, CP = C::CP, NESC = C::NESC, CSZ = C::CSZ;')
    s=once(s,'                    sx_repair(b[nt],sxh[nt],reinterpret_cast<const uint16_t*>(ptab),lane);',
        '                    if constexpr(C::STAGE) sxs_repair(b[nt],sxh[nt],s0+XBY+C::PAYLOAD+C::HEADERS,lane);\n'
        '                    else sx_repair(b[nt],sxh[nt],reinterpret_cast<const uint16_t*>(ptab),lane);')
    s=once(s,'        const int orow = C::SX ? ptab[pmax+rb*R+row] : (COMP ? tab[2 * pmax + row] : rb * R + row);',
        '        const int orow=C::STAGE?ptab[rb*R+row]:(C::SX?ptab[pmax+rb*R+row]:(COMP?tab[2*pmax+row]:rb*R+row));')
    s=once(s,'    if constexpr(C::SX) TORCH_CHECK(pmax>=0 && ptab.numel()>=(int64_t)pmax+N,"sx: patch/rowmap bounds");',
        '    if constexpr(C::SX) TORCH_CHECK(pmax>=0 && ptab.numel()>=(int64_t)(C::STAGE?0:pmax)+N,"sx: record/rowmap bounds");\n'
        '    if constexpr(C::STAGE) TORCH_CHECK(pmax<=C::NW*UPW*NT*512,"sxs: stage capacity bounds");')
    s=once(s,'    const size_t smem = (size_t)C::FIXED + tab_bytes;',
        '    const int sxs_wbytes=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);\n'
        '    const int sxs_sst=C::STAGE?((C::XBY+sxs_wbytes+1023)&~1023):C::SST;\n'
        '    const size_t smem=(size_t)C::FIXED+NSTAGE*(sxs_sst-C::SST)+tab_bytes;')
    return s


RUNTIME=r'''
def sxs_expected_layout(name,variant,pmax):
    assert variant in (0,1,2) and pmax>=0
    expected=xor_expected_layout(name,min(variant,1),pmax)
    if variant==2:
        expected=dict(expected)
        assert pmax<=_XOR_CONFIG[name]['geom']['NW']*_XOR_CONFIG[name]['geom']['UPW']*_XOR_CONFIG[name]['geom']['NT']*512
        records=(pmax*2+15)&~15
        wbytes=expected['weight_bytes']+records
        stride=(expected['x_reservation']+wbytes+1023)&~1023
        expected['dynamic']+=expected['stages']*(stride-expected['stage_stride'])
        expected.update(stage_stride=stride,weight_bytes=wbytes,record_bytes=records,stage_capacity=pmax)
    return expected

def sxs_build(tag):
    return load_inline(name='okc_owned_sxs_'+str(tag),cpp_sources=_SXS_CPP,cuda_sources=_SXS_CUDA,
        functions=['sk_fwd','sxs_resources'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],
        verbose=False,no_implicit_headers=True)

def sxs_repack(pk,name):
    # Derive a fixed arena from the actual maximum of the complete stage/CTA
    # population. No bounded-cap search, truncation or special-case omission.
    g=_XOR_CONFIG[name]['geom'];units=g['NW']*g['UPW']*g['NT']
    payload=units*768;headers=(units*4+15)&~15
    n,k=pk['n'],pk['k'];ns=(k//64)//g['UPS'];nc=n//g['R']
    old=pk['stream'].view(ns*nc,payload+headers)
    h=old[:,payload:payload+units*4].contiguous().view(torch.int32).to(torch.int64)&0xffffffff
    count=h&1023;start=h>>10
    assert bool((count<=512).all()),'SXS_UNIT_COUNT'
    local_start=count.cumsum(1)-count
    totals=count.sum(1);capacity=int(totals.max())
    assert capacity<=units*512,'SXS_CAPACITY'
    assert bool((local_start+count<=capacity).all()),'SXS_ARENA_BOUNDS'
    arena_bytes=(capacity*2+15)&~15
    flat_count=count.reshape(-1)
    expanded=torch.repeat_interleave(torch.arange(flat_count.numel(),device=old.device),flat_count)
    beginning=flat_count.cumsum(0)-flat_count
    rank=torch.arange(expanded.numel(),device=old.device)-beginning[expanded]
    source=start.reshape(-1)[expanded]+rank
    global_records=pk['table'][:pk['pmax']].view(torch.int16)
    assert bool((source<global_records.numel()).all()),'SXS_GLOBAL_RECORD_BOUNDS'
    # Pad only for the bulk-copy alignment. Local headers can never read padding.
    arena=torch.zeros(ns*nc,arena_bytes//2,dtype=torch.int16,device=old.device)
    destination=local_start.reshape(-1)[expanded]+rank
    if expanded.numel():arena[expanded//units,destination]=global_records[source]
    newh=((local_start<<10)|count).to(torch.int32).contiguous().view(torch.uint8)
    hp=torch.zeros(ns*nc,headers,dtype=torch.uint8,device=old.device);hp[:,:units*4]=newh
    stream=torch.cat((old[:,:payload],hp,arena.view(torch.uint8)),1).contiguous().view(-1)
    rowmap=pk['table'][pk['pmax']:].clone().contiguous()
    assert rowmap.numel()==n,'SXS_ROWMAP'
    return dict(stream=stream,table=rowmap,pmax=capacity,base4=pk['base4'],n=n,k=k,
        name=name,variant=2,immutable=[stream,rowmap],records=expanded.numel(),
        stage_capacity=capacity,stage_record_bytes=arena_bytes,stage_groups=ns*nc,
        header_bytes=ns*nc*headers,padded_record_bytes=ns*nc*arena_bytes,
        max_unit_count=int(count.max()),minimum_stage_count=int(totals.min()),
        global_source_record_words=pk['pmax'])

def sxs_pack(name,weight,variant):
    assert variant in (0,1,2)
    pk=xor_pack(name,weight,min(variant,1))
    if pk is None:return None
    return sxs_repack(pk,name) if variant==2 else pk

def sxs_runner(module,name,weight,variant,*,nearby=False,packed=None):
    pk=sxs_pack(name,weight,variant) if packed is None else packed
    if pk is None:return None
    assert pk['name']==name and pk['variant']==variant
    cfg=_SXS_CONFIG[name]['ids'][variant][int(nearby)]
    diag=torch.zeros(4,dtype=torch.int32,device=weight.device)
    def run(x):
        assert x.dtype==torch.bfloat16 and x.shape[1]==pk['k'] and x.is_contiguous()
        out=torch.empty((x.shape[0],pk['n']),dtype=torch.bfloat16,device=x.device)
        module.sk_fwd(cfg,pk['stream'],x,out,pk['table'],pk['pmax'],pk['base4'],pk['n'],pk['k'],diag)
        return out
    return run,dict(cfg=cfg,name=name,variant=variant,nearby=nearby,private_diag=diag,
        immutable=pk['immutable'],pack=pk,module_name=module.__name__,
        resources=list(module.sxs_resources(cfg,pk['pmax'])),expected=sxs_expected_layout(name,variant,pk['pmax']))
'''


def generate(raw):
    augmented,_,previous_manifest=previous.generate(raw)
    m=previous.load_c2(raw);originals={c[0]:c for c in m.SK_CONFIGS}
    configs=[];meta={}
    for name in NAMES:
        original=originals[name];ids=[]
        for variant in (0,1,2):
            pair=[]
            for nearby in (False,True):
                c=list(original);c[0]=name+'_sxs'+str(variant)+('_nf' if nearby else '')
                c[10]=(c[10]&~32 if nearby else c[10])|(4096 if variant else 0)|(8192 if variant==2 else 0)
                pair.append(len(configs));configs.append(tuple(c))
            ids.append(pair)
        meta[name]=dict(ids=ids,original_targs=m._sk_targs(original),original_cfg=original)
    m.SK_CONFIGS=configs;m.SK_INDEX={c[0]:i for i,c in enumerate(configs)}
    m._SK_SRC=patch_cuda(m._SK_SRC);cuda=m.sk_source();resource=[]
    for i,c in enumerate(configs):
        args=m._sk_targs(c).split(', ');ct=', '.join(args[:11]+args[12:]);nst=c[7]
        resource.append('case '+str(i)+':{using C=SkCfg<'+ct+'>;auto k=sk_kernel<'+m._sk_targs(c)+'>;\n'
            'const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);\n'
            'const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;\n'
            'const size_t bytes=C::FIXED+'+str(nst)+'*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);\n'
            'TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");\n'
            'cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));\n'
            'C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;\n'
            'C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));\n'
            'return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}')
    cuda+='\n#ifndef SK_DEVICE_ONLY\nstd::vector<int64_t> sxs_resources(int64_t cfg,int64_t pmax){switch(cfg){\n'+'\n'.join(resource)+'\ndefault:TORCH_CHECK(false,"sxs cfg");}}\n#endif\n'
    cpp='#include <torch/extension.h>\n'+m._SK_CPP+'\nstd::vector<int64_t> sxs_resources(int64_t cfg,int64_t pmax);\n'
    result=augmented+'\n# OFF-by-default stage-local record prototype.\n_SXS_CUDA = '+previous.visible_literal(cuda)+'\n_SXS_CPP = '+previous.visible_literal(cpp)+'\n_SXS_CONFIG = '+repr(meta)+'\n'+RUNTIME
    ast.parse(result)
    return result,'#define SK_DEVICE_ONLY 1\n'+cuda,dict(control_raw=PIN,global_prototype=previous_manifest['runtime_sha256'],
        bytes=len(result.encode()),sha256=hashlib.sha256(result.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        names=NAMES,gpu_execution=False,H200_qualified=False,capacity='maximum actual per-stage/CTA count, not tuned',normal_entry='unchanged C2')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result,cu,manifest=generate(a.control.read_bytes());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(result.encode())
    a.output.with_suffix('.cu').write_text(cu,encoding='utf8',newline='\n')
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest,indent=2))
