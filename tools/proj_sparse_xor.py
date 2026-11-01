"""OFF-by-default owned C2 SK sparse exact BF16 fragment repair prototype.

No judge dispatch. Original source remains the normal entry. New extension and
immutable formats are exposed only through xor_build/xor_pack/xor_runner.
"""
import argparse,ast,hashlib,importlib.util,json,types
from pathlib import Path

PIN='5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4'
BIT=4096
NAMES=('qkv8F','qkv16F','qkv32','o16F','o32F','pool16F','pool32F','unp16F','unp32F','slice8F')


def encode_record(lane,reg,half,xor7):
    assert 0<=lane<32 and 0<=reg<8 and half in (0,1) and 0<xor7<128
    return lane|(reg<<5)|(half<<8)|(xor7<<9)


def repair_bits(bits,base):
    assert 0<=bits<65536 and 0<=base<=120
    e=(bits>>8)&127;decoded=(bits&0x80ff)|((base+min(7,max(0,e-base)))<<8)
    return decoded,((bits^decoded)>>8)&127


def once(s,old,new):
    assert s.count(old)==1,(old[:100],s.count(old));return s.replace(old,new)


def visible_literal(text):
    for quote in ("'''",'"""'):
        if quote in text:continue
        value='r'+quote+text+quote
        try:
            if ast.literal_eval(value)==text:return value
        except SyntaxError:pass
    raise ValueError('source must stay a readable multiline literal')


HELPER=r'''
__device__ __forceinline__ uint32_t sx_lds32(uint32_t address) {
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(address) : "memory");
    return v;
}
// Header and all packed operands have been copied from the ring before release.
// Only immutable global records are read here. Every lane executes the same
// count/loop; lane ownership masks XOR without divergent MMA participation.
__device__ __forceinline__ void sx_repair(uint32_t (&b)[8], uint32_t h,
                                         const uint16_t* records, int lane) {
    const unsigned count=h&1023u, start=h>>10;
    for(unsigned i=0;i<count;++i){
        const unsigned rec=records[start+i];
        const unsigned reg=(rec>>5)&7u;
        const uint32_t x=((rec>>9)&127u)<<(8+16*((rec>>8)&1u));
        const uint32_t mask=(rec&31u)==(unsigned)lane ? x : 0u;
#pragma unroll
        for(int r=0;r<8;++r)b[r]^=reg==(unsigned)r ? mask : 0u;
    }
}
'''


def patch_cuda(s):
    # Default false: existing configurations compile their original dataflow.
    s=once(s,'// Eight 12-bit weights ->',HELPER+'\n// Eight 12-bit weights ->')
    s=once(s,'    static constexpr int NW = NWR * WSK;',
           '    static constexpr bool SX=(NACC & 4096)!=0;\n    static constexpr int NW = NWR * WSK;')
    s=once(s,'    static constexpr int WBY = NW * UPW * NT * TU;',
           '    static constexpr int PAYLOAD = NW * UPW * NT * TU;\n'
           '    static constexpr int HEADERS = SX ? ((NW*UPW*NT*4+15)&~15) : 0;\n'
           '    static constexpr int WBY = PAYLOAD + HEADERS;')
    s=once(s,'    static constexpr int NESC = COMP ? (ESCG ? 4 : (NSTAGE >= 4 ? 4 : NSTAGE)) : 0;',
           '    static constexpr int NESC = (COMP && !SX) ? (ESCG ? 4 : (NSTAGE >= 4 ? 4 : NSTAGE)) : 0;')
    s=once(s,'    static_assert(!(COMP && CL > 1),',
           '    static_assert(!SX || (COMP && !HL && CL==1 && !FUSE && !ESCG),"bounded sparse XOR");\n'
           '    static_assert(!(COMP && CL > 1),')
    s=once(s,'    const int tab_ints = COMP ? pmax * 2 + R : 0;',
           '    const int tab_ints = (COMP && !C::SX) ? pmax * 2 + R : 0;')
    # Only control escape table/correction users retain their barriers and work.
    # There are four COMP-only control regions plus the correction sum and map.
    s=s.replace('if constexpr (COMP) {','if constexpr (COMP && !C::SX) {')
    # Weight operand loads and decode are shared with control and MUST stay COMP.
    s=once(s,'                if constexpr (COMP && !C::SX) {\n                    lw[nt]',
           '                if constexpr (COMP) {\n                    lw[nt]')
    s=once(s,'            uint2 cw[NT];','            uint2 cw[NT];\n            uint32_t sxh[NT];')
    s=once(s,'                    cw[nt] = lds64(wb + nt * TU + 512 + lane * 8);',
           '                    cw[nt] = lds64(wb + nt * TU + 512 + lane * 8);\n'
           '                    if constexpr(C::SX) sxh[nt]=sx_lds32(s0+XBY+C::PAYLOAD+((warp*UPW+uu)*NT+nt)*4);')
    # Repair before loading A: original A fragments need not stay live across
    # sparse global reads. Every shared read still precedes the original release.
    anchor='            uint32_t a[4][MT][4];\n            load_a(xt, ui, a);'
    s=once(s,anchor,'''            if constexpr(C::SX) {
#pragma unroll
                for(int nt=0;nt<NT;++nt){
                    decode8(lw[nt].x,lw[nt].y,cw[nt].x,base4,&b[nt][0]);
                    decode8(lw[nt].z,lw[nt].w,cw[nt].y,base4,&b[nt][4]);
                    sx_repair(b[nt],sxh[nt],reinterpret_cast<const uint16_t*>(ptab),lane);
                }
            }
'''+anchor)
    s=once(s,'        const int orow = COMP ? tab[2 * pmax + row] : rb * R + row;',
           '        const int orow = C::SX ? ptab[pmax+rb*R+row] : (COMP ? tab[2 * pmax + row] : rb * R + row);')
    s=once(s,'    const int tab_bytes = COMP ? (pmax * 2 + C::R) * 4 : 0;',
           '    const int tab_bytes = (COMP && !C::SX) ? (pmax * 2 + C::R) * 4 : 0;\n'
           '    if constexpr(C::SX) TORCH_CHECK(pmax>=0 && ptab.numel()>=(int64_t)pmax+N,"sx: patch/rowmap bounds");')
    return s


RUNTIME=r'''
def xor_expected_layout(name,variant,pmax):
    assert name in _XOR_CONFIG and variant in (0,1) and pmax>=0
    c=_XOR_CONFIG[name]['original_cfg']
    _,mt,tokr,nt,nwr,wsk,upw,nstage,comp,hl,nacc,cl,outt,fuse=c
    assert comp and not hl and cl==1 and not fuse and not(nacc&16)
    nw=nwr*wsk;R=nwr*nt*8;ups=wsk*upw
    payload=nw*upw*nt*768;headers=((nw*upw*nt*4+15)&~15) if variant else 0
    xbytes=ups*tokr*128;stage=(xbytes+payload+headers+1023)&~1023
    nesc=0 if variant else (4 if nstage>=4 else nstage)
    correction=nesc*((16*mt*(R+1)+3)&~3)*4
    fixed=1024+nstage*stage+16*nstage+32+correction+128+16
    dynamic=fixed+(0 if variant else pmax*8+R*4)
    return dict(static=0,dynamic=dynamic,threads=(nw+1+nesc)*32,
        stage_stride=stage,weight_payload=payload,header_bytes=headers,
        weight_bytes=payload+headers,x_reservation=xbytes,stages=nstage,
        correction_bytes=correction,escape_readers=nesc,empty_arrivals=nw+nesc)

def xor_build(tag):
    return load_inline(name='okc_owned_sx_'+str(tag),cpp_sources=_XOR_CPP,
        cuda_sources=_XOR_CUDA,functions=['sk_fwd','sx_resources'],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],
        verbose=False,no_implicit_headers=True)

def xor_pack(name,weight,variant):
    assert name in _XOR_CONFIG and variant in (0,1)
    nrows=1024 if name=='slice8F' else weight.shape[0]
    g=_XOR_CONFIG[name]['geom'];R=g['R']
    # Original finite matrix/rejection domain. Bit-transport CPU proof is broader.
    if weight.dtype!=torch.bfloat16 or not bool(torch.isfinite(weight.float()).all()):return None
    if not variant:
        pk=pack_layer(weight,{name:g},{name:nrows},balance=name.startswith('qkv'))
        if pk is None:return None
        tab,pmax=pk['tabs'][R,nrows]
        return dict(stream=pk['streams'][name],table=tab,pmax=pmax,base4=pk['base4'],n=nrows,k=weight.shape[1],
                    immutable=[pk['streams'][name],tab],variant=0,name=name,escapes=pk['escapes'])
    bits=bf16_bits(weight);base=window_base(bits);_,escape,_=code_bits(bits,base)
    total=int(escape.sum())
    if total*MAX_ESCAPE_DENOM>bits.numel():return None
    perm=balance_perm(escape,R) if name.startswith('qkv') and total else torch.arange(weight.shape[0],device=weight.device)
    return _xor_pack_bits(name,bits,base,perm,nrows,total)

def _xor_pack_bits(name,bits,base,perm,nrows,total):
    # Exact bit-transport core. Public xor_pack alone enforces finite/sparse offer.
    g=_XOR_CONFIG[name]['geom'];R=g['R']
    bits=bits[perm][:nrows].contiguous();tv=tile_values(bits)
    _,_,decoded=code_bits(tv,base);bad=(tv!=decoded)
    tile,unit,lane,value=bad.nonzero(as_tuple=True)
    ntile,nu=tv.shape[:2];linear=tile*nu+unit
    count=torch.bincount(linear,minlength=ntile*nu).to(torch.int64)
    start=count.cumsum(0)-count
    if int(count.max())>512 or (len(linear) and int(start.max())>=(1<<22)):return None
    assert len(linear)<(1<<22),'SX_PATCH_OFFSET_OVERFLOW'
    xor=((tv[tile,unit,lane,value]^decoded[tile,unit,lane,value])>>8)&127
    records=(lane|((value//2)<<5)|((value%2)<<8)|(xor<<9)).to(torch.int16)
    # nonzero returns lexicographic tile/unit/lane/value order: same unit header
    # selects a unique finite list, no rank/half collision and no padding reader.
    pad=torch.zeros(((len(records)+1)//2)*2,dtype=torch.int16,device=bits.device)
    pad[:len(records)]=records
    patchwords=pad.view(torch.int32);pmax=patchwords.numel()
    table=torch.cat((patchwords,perm[:nrows].to(torch.int32))).contiguous()
    headers=((start<<10)|count).to(torch.int32).reshape(ntile,nu)
    payload=stage_stream(comp_units(bits,base),g,nrows)
    NW,UPW,NT,WSK=g['NW'],g['UPW'],g['NT'],g['WSK'];UPS=g['UPS']
    ns=nu//UPS;nc=nrows//R
    st=torch.arange(ns,device=bits.device).view(-1,1,1,1,1)
    c=torch.arange(nc,device=bits.device).view(1,-1,1,1,1)
    w=torch.arange(NW,device=bits.device).view(1,1,-1,1,1)
    u=torch.arange(UPW,device=bits.device).view(1,1,1,-1,1)
    nt=torch.arange(NT,device=bits.device).view(1,1,1,1,-1)
    tile_index=c*(R//8)+(w//WSK)*NT+nt;unit_index=st*UPS+(w%WSK)*UPW+u
    h=headers[tile_index,unit_index].contiguous().view(torch.uint8).reshape(ns,nc,-1)
    hbytes=(h.shape[-1]+15)&~15
    hp=torch.zeros(ns,nc,hbytes,dtype=torch.uint8,device=bits.device);hp[:,:,:h.shape[-1]]=h
    stream=torch.cat((payload.view(ns,nc,-1),hp),-1).contiguous().view(-1)
    return dict(stream=stream,table=table,pmax=pmax,base4=base*0x01010101,n=nrows,k=bits.shape[1],
        immutable=[stream,table],variant=1,name=name,escapes=total,records=len(records),
        header_bytes=hp.numel(),patch_bytes=patchwords.numel()*4,rowmap_bytes=nrows*4)

def xor_runner(module,name,weight,variant,*,nearby=False,packed=None):
    pk=xor_pack(name,weight,variant) if packed is None else packed
    if pk is None:return None
    assert pk['name']==name and pk['variant']==variant
    cfg=_XOR_CONFIG[name]['ids'][variant][int(nearby)]
    diag=torch.zeros(4,dtype=torch.int32,device=weight.device)
    def run(x):
        assert x.dtype==torch.bfloat16 and x.shape[1]==pk['k'] and x.is_contiguous()
        out=torch.empty((x.shape[0],pk['n']),dtype=torch.bfloat16,device=x.device)
        module.sk_fwd(cfg,pk['stream'],x,out,pk['table'],pk['pmax'],pk['base4'],pk['n'],pk['k'],diag)
        return out
    info=dict(cfg=cfg,name=name,variant=variant,nearby=nearby,private_diag=diag,immutable=pk['immutable'],
        pack=pk,module_name=module.__name__,resources=list(module.sx_resources(cfg,pk['pmax'])),
        expected=xor_expected_layout(name,variant,pk['pmax']))
    return run,info
'''


def load_c2(source):
    raw=source if isinstance(source,bytes) else Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PIN
    m=types.ModuleType('_sx_pinned_C2')
    exec(compile(raw.decode('utf8'),'<pinned-owned-C2>','exec'),m.__dict__)
    return m


def generate(source):
    raw=source if isinstance(source,bytes) else Path(source).read_bytes()
    m=load_c2(raw);byname={c[0]:c for c in m.SK_CONFIGS};configs=[];metadata={}
    for name in NAMES:
        original=byname[name];ids=[]
        for variant in (0,1):
            pair=[]
            for nearby in (False,True):
                c=list(original);c[0]=name+'_sx'+str(variant)+('_nf' if nearby else '')
                c[10]=(c[10]&~32 if nearby else c[10])|(BIT if variant else 0)
                pair.append(len(configs));configs.append(tuple(c))
            ids.append(pair)
        metadata[name]=dict(ids=ids,geom=m._sk_geom(name),original_targs=m._sk_targs(original),original_cfg=original)
    m.SK_CONFIGS=configs;m.SK_INDEX={c[0]:i for i,c in enumerate(configs)}
    m._SK_SRC=patch_cuda(m._SK_SRC);cuda=m.sk_source()
    resources=[]
    for i,c in enumerate(configs):
        args=m._sk_targs(c).split(', ');ct=', '.join(args[:11]+args[12:])
        resources.append('case '+str(i)+': { using C=SkCfg<'+ct+'>; auto k=sk_kernel<'+m._sk_targs(c)+'>;\n'
            'cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));\n'
            'C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;\n'
            'const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);\n'
            'C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));\n'
            'return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }')
    resource='\n#ifndef SK_DEVICE_ONLY\nstd::vector<int64_t> sx_resources(int64_t cfg,int64_t pmax){switch(cfg){\n'+'\n'.join(resources)+'\ndefault: TORCH_CHECK(false,"sx cfg");}}\n#endif\n'
    cuda+=resource
    cpp='#include <torch/extension.h>\n'+m._SK_CPP+'\nstd::vector<int64_t> sx_resources(int64_t cfg,int64_t pmax);\n'
    runtime='\n# Owned sparse E7-XOR prototype; normal C2 entry is unchanged.\n_XOR_CUDA = '+visible_literal(cuda)+'\n_XOR_CPP = '+visible_literal(cpp)+'\n_XOR_CONFIG = '+repr(metadata)+'\n'+RUNTIME
    normalized=raw.decode('utf8').replace('\r\n','\n')
    result=normalized+runtime
    ast.parse(result)
    return result,'#define SK_DEVICE_ONLY 1\n'+cuda,dict(source_pin=PIN,source_lf_sha256=hashlib.sha256(normalized.encode()).hexdigest(),names=NAMES,metadata=metadata,
        runtime_sha256=hashlib.sha256(result.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        bytes=len(result.encode()),gpu_execution=False,normal_entry='unchanged C2',status='prototype CPU/local compile gates pending')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    text,cu,manifest=generate(a.source);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(text.encode())
    a.output.with_suffix('.cu').write_text(cu,encoding='utf8',newline='\n')
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='metadata'},indent=2))
