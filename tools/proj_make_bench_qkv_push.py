"""Owned qkv128 push/local-reduction/vector-output epilogue, unchanged mainloop."""
from pathlib import Path
import ast
import sys
from proj_make_bench_qkv_depth import RUN,RESOURCE,ROOT

BIT=16777216
EPILOGUE=r'''
    if constexpr(C::QP){
        if(tid<C::NCT){
            // Cyclic eight-token fragments; peers push, own partial stays in registers.
#pragma unroll
            for(int j=0;j<BN/8;++j){
                const int dst=j%CLK;
                if(dst!=rk){
                    const int peer=rk-(rk>dst);
                    const uint32_t target=recv+((peer*C::QMAX+j/CLK)*C::NCT+tid)*16;
                    wd_st_async(wd_mapa(target,dst),make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]),wd_mapa(rbar,dst));
                }
            }
            mbar_wait_cluster(rbar,0,dead,diag,4);
            auto bf=reinterpret_cast<__nv_bfloat16*>(wd_smem+(sbase-raw));
#pragma unroll
            for(int j=0;j<BN/8;++j)if(j%CLK==rk){
                float4 v=make_float4(0.f,0.f,0.f,0.f);
#pragma unroll
                for(int p=0;p<CLK;++p){
                    float4 u;
                    if(p==rk)u=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
                    else u=wd_ld_local(recv+(((p-(p>rk))*C::QMAX+j/CLK)*C::NCT+tid)*16);
                    v.x+=u.x;v.y+=u.y;v.z+=u.z;v.w+=u.w;
                }
                // Pair adjacent output channels with a full-warp register exchange.
                const __nv_bfloat162 p0=__floats2bfloat162_rn(v.x,v.y),p8=__floats2bfloat162_rn(v.z,v.w);
                const uint32_t u0=*reinterpret_cast<const uint32_t*>(&p0),u8=*reinterpret_cast<const uint32_t*>(&p8);
                const uint32_t n0=__shfl_down_sync(0xffffffffu,u0,4),n8=__shfl_down_sync(0xffffffffu,u8,4);
                const int row=(warp&3)*16+(lane>>2),token=j*8+2*(lane&3);
                if((lane>>2)%2==0){
                    *reinterpret_cast<uint32_t*>(bf+token*72+row)=(u0&65535u)|(n0<<16);
                    *reinterpret_cast<uint32_t*>(bf+(token+1)*72+row)=(u0>>16)|(n0&0xffff0000u);
                    *reinterpret_cast<uint32_t*>(bf+token*72+row+8)=(u8&65535u)|(n8<<16);
                    *reinterpret_cast<uint32_t*>(bf+(token+1)*72+row+8)=(u8>>16)|(n8&0xffff0000u);
                }
            }
            // Generic shared writes are consumed across warps below.
            named_bar(1,C::NCT);
            const int own=(BN/8+CLK-1-rk)/CLK;
            for(int q=tid;q<own*64;q+=C::NCT){
                const int token=(rk+(q/64)*CLK)*8+(q%64)/8,col=(q%8)*8;
                if(tok0+token<M)*reinterpret_cast<uint4*>(out+(size_t)(tok0+token)*N+rowbase+col)=
                    *reinterpret_cast<const uint4*>(bf+token*72+col);
            }
        }
        // Every target observed its incoming tx bytes before any sender can exit.
        if constexpr(CL>1)wd_cluster_sync();
        return;
    }
'''

def literal(source,name):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))

def patch(src):
    original=src
    edits={
      '    static constexpr bool NOPROD =': '    static constexpr bool QP=(VAR & 16777216)!=0;\n    static constexpr int QMAX=(BN/8+CLK-1)/CLK;\n    static constexpr bool NOPROD =',
      '    static constexpr int SMEM = 1024 + RING + 16 * NST + (PUSH ? 16 + RECV : 0);':
      '    static constexpr int SMEM = 1024 + RING + 16 * NST + (PUSH ? 16 + RECV : (QP ? 16+(CLK-1)*QMAX*NCT*16 : 0));\n    static_assert(!QP || (WG==1 && BN==128 && CLK==3 && CLR==1 && !HL && NST==8 && !NOPROD),"bounded qkv push");',
      '        if constexpr (PUSH) {\n            mbar_init(rbar, 1);':
      '        if constexpr (PUSH || C::QP) {\n            mbar_init(rbar, 1);',
      '        fence_mbar_init();\n        if constexpr (PUSH) {':
      '        fence_mbar_init();\n        if constexpr(C::QP){mbar_expect_tx(rbar,(CLK-1)*((BN/8+CLK-1-rk)/CLK)*C::NCT*16);}\n        if constexpr (PUSH) {',
    }
    for old,new in edits.items():
        assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
    a=src.index('            const int row = g * 64 + (warp & 3) * 16 + (lane >> 2);',src.index('        if constexpr (!PUSH) {'))
    b=src.index('\n        }\n    }\n\n    if constexpr (PUSH) {',a)
    oldpartial=src[a:b];src=src[:a]+'            if constexpr(!C::QP){\n'+oldpartial+'\n            }'+src[b:]
    pos=src.index("    // Also keeps a CTA's ring alive")
    at=src.index('    if constexpr (C::GRED &&',pos)
    src=src[:at]+EPILOGUE+'\n'+src[at:]
    start=original.index('    float acc[NACC];');end=original.index('        if constexpr (!PUSH) {',start)
    assert original[start:end]==src[src.index('    float acc[NACC];'):src.index('        if constexpr (!PUSH) {',src.index('    float acc[NACC];'))]
    return src

def cpu_gate():
    # Every incoming destination slot has one peer writer and exactly the expected bytes.
    for dst in range(3):
        slots=[]
        for peer in range(3):
            if peer==dst:continue
            for j in range(16):
                if j%3==dst:
                    for tid in range(128):slots.append(((peer-(peer>dst))*6+j//3)*128+tid)
        assert len(slots)==len(set(slots))==2*((18-dst)//3)*128
        assert min(slots)>=0 and max(slots)<2*6*128
        # Simulate all four fragment components and adjacent-channel pair packing.
        buf={}
        for j in range(16):
            if j%3!=dst:continue
            for tid in range(128):
                warp,lane=divmod(tid,32);row=warp*16+lane//4;token=j*8+2*(lane%4)
                if (lane//4)%2==0:
                    for rr in (row,row+8):
                        for t in (token,token+1):
                            for r in (rr,rr+1):
                                off=t*72+r;assert off not in buf;buf[off]=(t,r)
        expected={(j*8+t,r) for j in range(16) if j%3==dst for t in range(8) for r in range(64)}
        assert set(buf.values())==expected
        for m in (100,127,128,129):
            for tok0 in range(0,m,128):
                output=[]
                for q in range(((18-dst)//3)*64):
                    t=(dst+(q//64)*3)*8+(q%64)//8;c=(q%8)*8
                    assert (t*72+c)*2%16==0 and t*72+c+8<=128*72
                    if tok0+t<m:output.extend((tok0+t,c+r) for r in range(8))
                assert len(output)==len(set(output))
                assert set(output)=={(tok0+t,r) for t,r in expected if tok0+t<m}
    assert 197760+16+2*6*128*16==222352<227*1024

def make(source):
    wd=patch(literal(source,'_WD_SRC'))
    runtime=RUN.replace('_qd','_qp').replace('QD','QP').replace('qdepth','qpush').replace('okc_qkv_depth','okc_qkv_push')
    runtime=runtime.replace('8388608','16777216').replace('_d1','_pull').replace('_d2','_push')
    runtime=runtime.replace('depth1','pull').replace('depth2','push').replace('d2-d1','push-pull')
    resource=RESOURCE.replace('8388608','16777216').replace('qdepth','qpush')
    result=source+runtime.replace('@@SOURCE@@',repr(wd)).replace('@@RESOURCE@@',repr(resource))
    ast.parse(result);return result

def host_gate(source):
    ns={};keys=('SK_CONFIGS','SK_INDEX','WD_GEOM','WD_VAR','WD_NAMES','WD_INDEX','WD_ABL','WD_POL','_WD_SRC')
    saved={k:object() for k in keys};ns.update(saved)
    ns.update(_SK_CPP='',_WD_CPP='',sk_source=lambda:'sk',wd_source=lambda:'wd',load_inline=lambda **kw:kw)
    tree=ast.parse(make(source));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_qpbuild')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual builder>','exec'),ns)
    result=ns['_qpbuild']('cpu_gate')
    assert result['cuda_sources'].startswith('skwd') and all(ns[k] is saved[k] for k in keys)

if __name__=='__main__':
    source=(ROOT/'muse_bf16_proj/submission.py').read_text(encoding='utf-8')
    cpu_gate();host_gate(source);Path(sys.argv[1]).write_text(make(source),encoding='utf-8')
    print('CPU peer slots/tx counts/shared transpose/output bounds/host restore/unchanged mainloop PASS')
