"""Fixed producer-decoded shared-A pipeline, ordinary layer candidate; no dispatch."""
import argparse
import ast
import difflib
import json
import re
from pathlib import Path
from g_expert_g224_prefix import BASE,BASE_SHA,NATIVE_SHA,sha,once,literal,function_source

ROOT=Path(__file__).resolve().parents[1]


def ss_mma():
    out=['template <int NTT> struct WGSMMA;']
    for n in range(1,9):
        out.append(f'template <> struct WGSMMA<{n}> {{')
        for name,scale,constraint in (('mma0',0,'=f'),('mma',1,'+f')):
            outputs=', '.join('%'+str(i) for i in range(n*4))
            operands=', '.join(f'"{constraint}"(d[{i//4}][{i%4}])' for i in range(n*4))
            out.append(f'''  static __device__ __forceinline__ void {name}(float (&d)[WGS_NT][4], uint64_t ad, uint64_t xd) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{n*8}k16.f32.bf16.bf16 {{{outputs}}}, %{n*4}, %{n*4+1}, {scale}, 1, 1, 0, 0;\\n"
      : {operands} : "l"(ad), "l"(xd));
  }}''')
        out.append('};')
    return '\n'.join(out)+'\n'


SSA_MMA=ss_mma()
SSA_HELPERS=r'''
constexpr int SSA_AB = 256 * 128 * 2;
constexpr int SSA_AOFF = 2 * WGS_STG, SSA_XOFF = SSA_AOFF + 2 * SSA_AB;
__device__ __forceinline__ uint32_t ssa_boundary(uint32_t total,int c,int grid) {
  return (uint32_t)((unsigned long long)total*(unsigned)c/(unsigned)grid);
}
__device__ __forceinline__ int ssa_f(int row,int k) {
  return (row >> 3)*1024+(k >> 3)*64+(row & 7)*8+(k & 7);
}
__device__ __forceinline__ int ssa_gs(const Smem& S,int n,int& pb) {
  const int na=S.wgr[1]-S.wgr[0];
  pb=n>=na;
  const int i=pb ? n-na : n,g0=S.wgr[2*pb],g1=S.wgr[2*pb+1];
  int gs=wgs_split(pb,g0,g1)+i;
  if(gs>=g1)gs=g0+gs-g1;
  return gs;
}
__device__ __forceinline__ void ssa_raw_issue(const Params& p,Smem& S,unsigned char* ring,int n) {
  const int slot=n&1;const uint32_t phase=(uint32_t)(n>>1)&1u;
  if(n>=2)wgs_wait(S.bars+BW_E+slot,phase^1u,1000+n);
  int pb;const int gs=ssa_gs(S,n,pb),ubl=pb ? 4 : 5;
  const int u=gs>>5,st=gs&31,j=u>>ubl,ub=u&((1<<ubl)-1);
  const unsigned char* base=pb ? p.wBs : p.wAs;
  const unsigned char* src=base+(size_t)(((S.jE[j]<<ubl)+ub)*32+st)*WGS_STG;
  uint64_t* bar=S.bars+BW_F+slot;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),"r"(WGS_STG):"memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
    ::"r"(smem_u32(ring+slot*WGS_STG)),"l"(src),"r"(WGS_STG),"r"(smem_u32(bar)) L2H_ARG :"memory");
}
'''

SSA_PRODUCER=r'''
__device__ void wgs_wproducer(const Params& p,Smem& S,unsigned char* ring) {
  WS_JOIN();
  const int dt=threadIdx.x-NT,wp=dt>>5,lane=dt&31;
  const int total=S.wgr[1]-S.wgr[0]+S.wgr[3]-S.wgr[2];
  if(dt==0){if(total>0)ssa_raw_issue(p,S,ring,0);if(total>1)ssa_raw_issue(p,S,ring,1);}
#pragma unroll 1
  for(int n=0;n<total;++n){
    const int slot=n&1;const uint32_t phase=(uint32_t)(n>>1)&1u;
    wgs_wait(S.bars+BW_F+slot,phase,3000000+n);
    if(n>=2)wgs_wait(S.bars+BA_E+slot,phase^1u,6000000+n);
    __nv_bfloat16* a=reinterpret_cast<__nv_bfloat16*>(ring+SSA_AOFF+slot*SSA_AB);
    const uint32_t k16c=16u+p.zero;
#pragma unroll
    for(int target=0;target<4;++target){
      const unsigned char* pc=ring+slot*WGS_STG+target*WGS_PIECE;
      uint32_t scw;
#if OKC_SC6W
      const uint32_t packed=(uint32_t)pc[4096+dt]|((uint32_t)pc[4096+128+dt]<<8)|((uint32_t)pc[4096+256+dt]<<16);
      scw=((packed&63u)+64u)|((((packed>>6)&63u)+64u)<<8)|((((packed>>12)&63u)+64u)<<16)|((((packed>>18)&63u)+64u)<<24);
#else
      scw=*reinterpret_cast<const uint32_t*>(pc+4096+dt*4);
#endif
#pragma unroll
      for(int lg=0;lg<4;++lg){
        const uint2 cw=*reinterpret_cast<const uint2*>(pc+((lg*4+wp)*32+lane)*8);
#pragma unroll
        for(int js=0;js<2;++js){
          const uint32_t q=js ? cw.y : cw.x;
          uint32_t r0,r1,r2,r3;dq4(q,r0,r1,r2,r3);
          const uint32_t sb=(uint32_t)js*2u;
          const uint32_t sgp=__byte_perm(scw,0u,0x4040u|sb|((sb+1u)<<8))*k16c+0x5C005C00u;
          const int row=target*64+wp*16+(lane>>2),k=(lg*2+js)*16+2*(lane&3);
          *reinterpret_cast<uint32_t*>(a+ssa_f(row,k))=bmul2l(r0,sgp);
          *reinterpret_cast<uint32_t*>(a+ssa_f(row+8,k))=bmul2h(r1,sgp);
          *reinterpret_cast<uint32_t*>(a+ssa_f(row,k+8))=bmul2l(r2,sgp);
          *reinterpret_cast<uint32_t*>(a+ssa_f(row+8,k+8))=bmul2h(r3,sgp);
        }
      }
    }
    // Every writer publishes its own generic stores to the async proxy.
    // Raw reads have also retired before all128 arrivals release that slot.
    asm volatile("fence.proxy.async.shared::cta;\n" :::"memory");
    mbar_arrive1(S.bars+BA_F+slot);
    mbar_arrive1(S.bars+BW_E+slot);
    asm volatile("bar.sync 8, 128;\n" :::"memory");
    if(dt==0 && n+2<total)ssa_raw_issue(p,S,ring,n+2);
  }
}
'''

SSA_SEG=r'''
  const uint32_t xring_s=smem_u32(ring+SSA_XOFF);
  const uint32_t aring_s=smem_u32(ring+SSA_AOFF+wg*16384);
  const uint64_t xd0=wg_desc(xring_s,128,2048),ad0=wg_desc(aring_s,128,2048);
  float d[WGS_NT][4];
  auto stage=[&](auto first){
    constexpr bool FIRST=decltype(first)::value;
    const int slot=seq&1;const uint32_t phase=(uint32_t)(seq>>1)&1u;
    wgs_wait(S.bars+BA_F+slot,phase,3000000+seq);
    wgs_wait(S.bars+BX_F+slot,phase,4000000+seq);
    const uint64_t ad=ad0+(uint64_t)(uint32_t)(slot*(SSA_AB>>4));
    const uint64_t xd=xd0+(uint64_t)(uint32_t)(slot*(WGS_XS>>4));
#pragma unroll
    for(int ci=0;ci<4;++ci){
      wg_fence();
#pragma unroll
      for(int jj=0;jj<2;++jj){
        const uint64_t off=(uint64_t)((ci*2+jj)*16);
        if(FIRST && ci==0 && jj==0)WGSMMA<NTT>::mma0(d,ad,xd);
        else WGSMMA<NTT>::mma(d,ad+off,xd+off);
      }
      wg_commit();wg_wait<1>();
      if(FIRST && ci==3 && pd.kind)wgs_flush<PB>(p,S,pd);
    }
    // All512 consumers retire both async operand reads before slot reuse.
    wg_wait<0>();
    asm volatile("fence.proxy.async.shared::cta;\n" :::"memory");
    mbar_arrive1(S.bars+BA_E+slot);
    mbar_arrive1(S.bars+BX_E+slot);
    ++seq;
  };
'''


def transform(original):
    assert sha(original)==NATIVE_SHA
    a=original.index('#define OKC_WGSNX 3');b=original.index('\n#endif\n\ntemplate <bool PB, bool BIG>',a)
    r=original[a:b]
    r=once(r,'#define OKC_WGSNX 3','#define OKC_WGSNX 2')
    r=once(r,'constexpr int WGS_NS = 3;','constexpr int WGS_NS = 2;')
    r=once(r,'constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;','constexpr int WGS_PIECE = OKC_SC6W ? 4480 : 4608;')
    r=once(r,'constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;','constexpr int WGS_STG = 4 * WGS_PIECE;')
    r=once(r,'constexpr int WGS_XS = WGS_NT * 4096;','constexpr int WGS_XS = WGS_NT * 2048;')
    r=once(r,'constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;','constexpr int BW_F=0,BW_E=2,BA_F=4,BA_E=6,BX_F=8,BX_E=10;')
    r=once(r,'static_assert(BX_E + WGS_NX <= 24, "WGS barriers");','static_assert(BX_E+WGS_NX<=24 && NT==512 && NTL==672 && FT_T==64 && WGS_NS==2 && WGS_NX==2,"SSA fixed geometry");')
    r=r.replace('+ 136 * 16;','+ JMAX * 32;').replace('MF_DA = 384, MF_INTS = 4096','MF_DA = 512, MF_INTS = 8192')
    r=r.replace('JMAX * 16','JMAX * 32').replace('j * 16 +','j * 32 +')
    r=once(r,'j * 32 + st) < 8','j * 32 + st) < 4')
    r=r.replace('(ub >> 1)','ub')
    for old,new in [('gs >> 4','gs >> 5'),('gs & 15','gs & 31'),('g1 & 15','g1 & 31'),('g0 & 15','g0 & 31'),('((g1 - 1) >> 4) << 4','((g1 - 1) >> 5) << 5'),('s1 - s0 == 16','s1 - s0 == 32'),('s1 == 16','s1 == 32'),('min(16,','min(32,'),('16 - pd.nst','32 - pd.nst'),('old + pd.nst == 16','old + pd.nst == 32'),('u << 4, gridDim.x','u << 5, gridDim.x'),('16u * cs','32u * cs'),('min(15u,','min(31u,'),('un * 16u','un * 32u'),('9 - pb','10 - pb'),('9 - pbl','10 - pbl'),('(sj >> 4)','(sj >> 5)'),('(sj & 15u)','(sj & 31u)'),('wgs_cost(0, S.jC[j]) * 16u','wgs_cost(0, S.jC[j]) * 32u'),('wgs_cost(1, S.jC[j]) * 16u','wgs_cost(1, S.jC[j]) * 32u')]:
        r=r.replace(old,new)
    for index in ('c','c + 1','c + kk'):
        r=r.replace('tot * (uint32_t)'+('('+index+')' if '+' in index else index)+' / (uint32_t)G',f'ssa_boundary(tot, {index}, G)')
    x=r.index('template <int NTT> struct WGSMMA;');y=r.index('// two producer warps:',x) if '// two producer warps:' in r[x:] else r.index('// two producer warps',x)
    r=r[:x]+SSA_MMA+r[y:]
    x=r.index('__device__ void wgs_wproducer(');y=r.index('// x producer:',x)
    r=r[:x]+SSA_HELPERS+SSA_PRODUCER+'\n'+r[y:]
    r=once(r,'unsigned char* xring = ring + WGS_NS * UNIT_BYTES;','unsigned char* xring = ring + SSA_XOFF;')
    r=r.replace('npad * 512u','npad * 256u').replace('st * npad * 256','st * npad * 128')
    x=r.index('__device__ void wgs_gather(');y=r.index('// Work left from the previous segment',x)
    gather=r[x:y]
    gather=gather.replace('tid >> 5','tid >> 4').replace('tid & 31','tid & 15').replace(' * 32 + (tid & 15)',' * 16 + (tid & 15)').replace('(tid >> 4) * 256','(tid >> 4) * 128')
    gather=gather.replace('8 * ((G >> 1) & 3) + 2 * (G >> 3)','8 * ((G >> 1) & 1) + 2 * (G >> 2)')
    gather=gather.replace('xs32 + 32','xs32 + 16').replace('xs32 + 64','xs32 + 32').replace('xs32 + 96','xs32 + 48')
    r=r[:x]+gather+r[y:]
    r=r.replace('col >> 8','col >> 7').replace('col & 255','col & 127').replace('* 32 + ((col & 127)','* 16 + ((col & 127)')
    x=r.index('  const uint32_t k16c =');y=r.index('  stage(std::true_type{});',x)
    r=r[:x]+SSA_SEG+r[y:]
    r=r.replace('one warpgroup\'s weights in a stage (SC6: 8 KB codes + 768 B scales)','one WG compact weights (SC6: 4 KB codes + 384 B scales)')
    r=r.replace('weight ring slots (36 KB; freed as soon as the codes are in registers)','compact raw slots; released by all128 decoder readers')
    r=r.replace('bytes of a stage in global memory (the smem slot keeps UNIT_BYTES)','compact bytes per global/shared raw stage')
    r=r.replace('an x slot: up to FT_T rows x 256 K','an x slot: up to FT_T rows x128 K')
    r=r.replace('// phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them','// Phase B K128 stage st reads one phase-A unit: four WG publications at index st.')
    r=r.replace('// acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2','// acquire. Unit ub wrote all128 columns of phase-B stage ub.')
    r=r.replace('// kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,','// kind3 finishes the unit: phase A share holding stage31,')
    r=r.replace('// this CTA\'s stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same','// This CTA\'s stage ranges:256-row units x32 stages; all producers use the same ordering.')
    r=r.replace('#define OKC_WGSFCI 4  // st2 (E374): the deferred flush at commit group 4 (FCI 3..7 sweep best)','#define OKC_WGSFCI 4  // legacy RS macro unused by SSA: the fixed SS flush is ci3')
    changed=original[:a]+r+original[b:]
    changed=once(changed,'// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)','// ---------------------------------------------------------------- WGS: producer-decoded SS-wgmma (row split, stream-K)')
    changed=once(changed,'// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,','// Unit=(job,256rows):32 units/job in A,16 in B; each has32 stages of128K with four compact WG pieces.')
    changed=once(changed,'NTL = NT + (OKC_WGS ? 64','NTL = NT + (OKC_WGS ? 160')
    changed=once(changed,'if (tid < NT + 32) wgs_wproducer','if (tid < NT + 128) wgs_wproducer')
    changed=once(changed,'for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT);','for(int k=0;k<BX_E+WGS_NX;++k)mbar_init(S.bars+k,k<BW_E ? 1 : k<BA_F ? 128 : k<BA_E ? 128 : k<BX_F ? NT : k<BX_E ? 1 : NT);')
    changed=changed.replace('JMAX * 16','JMAX * 32')
    changed=once(changed,'return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;','return base + okc::SSA_XOFF + (size_t)okc::WGS_NX * okc::WGS_XS;')
    changed=once(changed,'#include <torch/extension.h>','#include <torch/extension.h>\n#include <cstdio>')
    changed=once(changed,'  void* args[]={&p};', '''  static unsigned admission_seen=0;
  const unsigned bit=p.M>=OKC_BIGM ? 2u : 1u;
  if(!(admission_seen&bit)){
    cudaFuncAttributes actual{};
    TORCH_CHECK(cudaFuncGetAttributes(&actual,kernel)==cudaSuccess,"SSA_RESOURCE attributes");
    std::printf("SSA_ADMISSION b4=%d big=%d m=%d sm=%d grid=%d threads=%d dynamic=%zu static=%zu local=%zu regs=%d resident=%d\\n",
      OKC_B4FIN,int(p.M>=OKC_BIGM),p.M,prop.multiProcessorCount,grid,okc::NTL,
      sm,actual.sharedSizeBytes,actual.localSizeBytes,actual.numRegs,resident);
    std::fflush(stdout);admission_seen|=bit;
  }
  void* args[]={&p};''')
    return changed


def packers():
    # Only the independently checked K128 permutation is reused, not the parked
    # two-WG geometry, dispatch, finalizer, scratch or device transform.
    from wgs_k128_full_reuse import PACKERS
    s=PACKERS.replace('rows.numel() // 128','rows.numel() // 256')
    s=s.replace('view(nub, 2, 4, 32','view(nub, 4, 4, 32').replace('(nub, 32, 2,','(nub, 32, 4,')
    s=s.replace('view(nub, 2, 4, 2, 8','view(nub, 4, 4, 2, 8')
    return s


APPENDIX='''
_SSA_PARENT_EXT=_ma_ext
_SSA_EXTS={}
def _ma_ext(defines=None):
    defines=tuple(MOE_DEFINES if defines is None else defines)
    if "-DOKC_WGS=1" not in defines:return _SSA_PARENT_EXT(defines)
    if any(v.startswith("@") for v in defines):raise RuntimeError("SSA alternate WGS source not allocated")
    if DQ7 and "-DOKC_DQ7=1" not in defines:defines+=("-DOKC_DQ7=1",)
    if defines not in _SSA_EXTS:
        if torch.cuda.get_device_capability()!=(9,0):raise RuntimeError("SSA requires SM90")
        import hashlib
        tag=hashlib.sha256((_SSA_CUDA+repr(defines)).encode()).hexdigest()[:16]
        _SSA_EXTS[defines]=load_inline(name="shared_a_"+tag,cpp_sources=MOE_CPP_SRC,cuda_sources=_SSA_CUDA,
            functions=["okc_run","okc_set_pdl","okc_attrs"],extra_cflags=["-std=c++20"],
            extra_cuda_cflags=["-O3","-std=c++20","-gencode=arch=compute_90a,code=sm_90a","-lineinfo","-DOKC_NS=3","-DOKC_FNST=4","-DOKC_NSF=3",*defines],verbose=False)
        _SSA_EXTS[defines].okc_set_pdl(0)
    return _SSA_EXTS[defines]

def _ma_make_scratch(dev):
    jobs=32+_MA_EMAX//64+8
    fixup=2*160*4*4096+2*160*4+jobs*32
    return [torch.zeros(8+1024+_MA_EMAX+8192,dtype=torch.int32,device=dev),
            torch.zeros(_MA_EMAX+_MA_EMAX*4096+fixup,dtype=torch.float32,device=dev),
            _df_inter(torch.zeros((_MA_EMAX+2*WGS_XROWS)*4096,dtype=torch.bfloat16,device=dev))]

def _fork_install_fallback(weights,state,reason):
    raise RuntimeError("SSA_STOP no silent fallback: "+reason)
'''


def generate(out):
    base=(ROOT/BASE).read_text();assert sha(base)==BASE_SHA
    original=literal(base,'MOE_CUDA_SRC');native=transform(original)
    parent=(ROOT/'tools/g_expert_g224_prefix.py').read_text()
    own=Path(__file__).read_text()
    fn=function_source(own,'transform').replace('def transform(','def _ssa_transform(')
    fn=fn.replace('sha(original)==NATIVE_SHA',f'_ssa_hashlib.sha256(original.encode()).hexdigest()=={NATIVE_SHA!r}').replace('once(','_ssa_once(')
    suffix='\nimport hashlib as _ssa_hashlib\n'+function_source(parent,'once').replace('def once(','def _ssa_once(')+'\n'
    for key,value in [('SSA_MMA',SSA_MMA),('SSA_HELPERS',SSA_HELPERS),('SSA_PRODUCER',SSA_PRODUCER),('SSA_SEG',SSA_SEG)]:suffix+=key+'='+repr(value)+'\n'
    suffix+=fn+'\n_SSA_CUDA=_ssa_transform(MOE_CUDA_SRC)\n'+packers()+APPENDIX
    candidate=base+suffix;ast.parse(candidate);assert len(candidate.encode())<1_000_000
    out.mkdir(parents=True,exist_ok=True)
    for name,text in [('candidate.py',candidate),('selected.cu',native),('native.diff',''.join(difflib.unified_diff(original.splitlines(True),native.splitlines(True),fromfile='ba9-c7-original',tofile='shared-a-fixed')))]:
        (out/name).write_text(text,encoding='utf8',newline='\n')
    m=dict(base_sha256=BASE_SHA,control_native_sha256=NATIVE_SHA,native_sha256=sha(native),candidate_sha256=sha(candidate),candidate_bytes=len(candidate.encode()),gpu=False,dispatch=False)
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return candidate,native,m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(generate(a.out)[2],indent=2))
