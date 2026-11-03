"""Fixed original-K256 direct-global register-A delivery; no dispatch/default edits."""
import argparse
import ast
import difflib
import json
from pathlib import Path
from g_expert_g224_prefix import BASE,BASE_SHA,NATIVE_SHA,sha,once,literal,function_source

ROOT=Path(__file__).resolve().parents[1]

ADDRESS=r'''
__device__ __forceinline__ size_t dw_weight_offset(int expert,int ub,int physical_st,int wg,int ubl) {
  return ((size_t)((expert << ubl)+ub)*16u+(size_t)physical_st)*(size_t)WGS_STG+(size_t)wg*WGS_PIECE;
}
'''


def transform(original):
    assert sha(original)==NATIVE_SHA
    start=original.index('#define OKC_WGSNX 3')
    end=original.index('\n#endif\n\ntemplate <bool PB, bool BIG>',start)
    r=original[start:end]
    line=next(x for x in r.splitlines(True) if x.startswith('constexpr int WGS_NS ='))
    r=once(r,line,'')
    r=once(r,'constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;','constexpr int BX_F=0,BX_E=WGS_NX;')
    r=once(r,'static_assert(BX_E + WGS_NX <= 24, "WGS barriers");','static_assert(BX_E+WGS_NX<=24 && NT==512 && NTL==544 && FT_T==64 && WGS_NX==3,"direct weight fixed geometry");')
    a=r.index('__device__ void wgs_wproducer(');b=r.index('// x producer:',a)
    r=r[:a]+r[b:]
    r=once(r,'unsigned char* xring = ring + WGS_NS * UNIT_BYTES;','unsigned char* xring = ring;')
    r=once(r,'// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)',ADDRESS+'\n// Original K256 RS segment; global weights use physical stage, X uses seq parity.')
    r=once(r,'const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);','const uint32_t xring_s = smem_u32(ring);')
    r=once(r,'const unsigned char* pc0 = ring + wg * WGS_PIECE;','const unsigned char* pc0 = (PB ? p.wBs : p.wAs) + dw_weight_offset(S.jE[u >> UBL],u & ((1 << UBL)-1),0,wg,UBL);')
    r=once(r,'auto stage = [&](auto first) {','auto stage = [&](auto first,int physical_st) {')
    r=once(r,'const int ws = seq % WGS_NS, xs = seq % WGS_NX;','const int xs = seq % WGS_NX;')
    r=once(r,'    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n','')
    r=once(r,'const unsigned char* pc = pc0 + ws * UNIT_BYTES;','const unsigned char* pc = pc0 + (size_t)physical_st * WGS_STG;')
    xwait='    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n'
    r=once(r,xwait,'')
    anchor='''#endif
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {'''
    r=once(r,anchor,'#endif\n'+xwait+'#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {')
    raw_release='''      } else {
        asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }'''
    r=once(r,raw_release,'      }')
    r=once(r,'stage(std::true_type{});','stage(std::true_type{},s0);')
    r=once(r,'stage(std::false_type{});','stage(std::false_type{},st);')
    r=r.replace('// bytes of a stage in global memory (the smem slot keeps UNIT_BYTES)','// immutable global stage bytes; no raw shared allocation')
    old_producers='''// two producer warps: the weights warp streams every stage's 36 KB through the weight ring (TMA, lane 0), the x warp
// gathers every stage's x slice into the x ring (cp.async) and, in phase B, first waits for the job's phase-A units;
// so the weights run ahead by the whole ring even while x waits'''
    r=once(r,old_producers,'// Sole X producer retains its three slots. Consumers fetch immutable weights directly into register A.')
    changed=original[:start]+r+original[end:]
    changed=once(changed,'NTL = NT + (OKC_WGS ? 64','NTL = NT + (OKC_WGS ? 32')
    role='''    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));'''
    changed=once(changed,role,'    wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));')
    init='for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT);'
    changed=once(changed,init,'for(int k=0;k<BX_E+WGS_NX;++k)mbar_init(S.bars+k,k<BX_E ? 1 : NT);')
    changed=once(changed,'uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty','uint64_t bars[24];  // DW:0..2 Xfull,3..5 Xempty; retain original header size')
    changed=once(changed,'return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;','return base + (size_t)okc::WGS_NX * okc::WGS_XS;')
    changed=once(changed,'#include <torch/extension.h>','#include <torch/extension.h>\n#include <cstdio>')
    changed=once(changed,'  void* args[]={&p};', '''  static unsigned admission_seen=0;
  const unsigned bit=p.M>=OKC_BIGM ? 2u : 1u;
  if(!(admission_seen&bit)){
    cudaFuncAttributes actual{};
    TORCH_CHECK(cudaFuncGetAttributes(&actual,kernel)==cudaSuccess,"DW_RESOURCE attributes");
    std::printf("DW_ADMISSION b4=%d big=%d m=%d sm=%d grid=%d threads=%d dynamic=%zu static=%zu local=%zu regs=%d resident=%d\\n",
      OKC_B4FIN,int(p.M>=OKC_BIGM),p.M,prop.multiProcessorCount,grid,okc::NTL,
      sm,actual.sharedSizeBytes,actual.localSizeBytes,actual.numRegs,resident);
    std::fflush(stdout);admission_seen|=bit;
  }
  void* args[]={&p};''')
    return changed


APPENDIX='''
_DW_PARENT_EXT=_ma_ext
_DW_EXTS={}
def _ma_ext(defines=None):
    defines=tuple(MOE_DEFINES if defines is None else defines)
    if "-DOKC_WGS=1" not in defines:return _DW_PARENT_EXT(defines)
    if any(v.startswith("@") for v in defines):raise RuntimeError("DW alternate WGS source not allocated")
    if DQ7 and "-DOKC_DQ7=1" not in defines:defines+=("-DOKC_DQ7=1",)
    if defines not in _DW_EXTS:
        if torch.cuda.get_device_capability()!=(9,0):raise RuntimeError("DW requires SM90")
        import hashlib
        tag=hashlib.sha256((_DW_CUDA+repr(defines)).encode()).hexdigest()[:16]
        _DW_EXTS[defines]=load_inline(name="direct_weight_"+tag,cpp_sources=MOE_CPP_SRC,cuda_sources=_DW_CUDA,
            functions=["okc_run","okc_set_pdl","okc_attrs"],extra_cflags=["-std=c++20"],
            extra_cuda_cflags=["-O3","-std=c++20","-gencode=arch=compute_90a,code=sm_90a","-lineinfo","-DOKC_NS=3","-DOKC_FNST=4","-DOKC_NSF=3",*defines],verbose=False)
        _DW_EXTS[defines].okc_set_pdl(0)
    return _DW_EXTS[defines]

def _fork_install_fallback(weights,state,reason):
    raise RuntimeError("DW_STOP no silent fallback: "+reason)
'''


def generate(out):
    base=(ROOT/BASE).read_text();assert sha(base)==BASE_SHA
    original=literal(base,'MOE_CUDA_SRC');native=transform(original)
    own=Path(__file__).read_text();parent=(ROOT/'tools/g_expert_g224_prefix.py').read_text()
    fn=function_source(own,'transform').replace('def transform(','def _dw_transform(').replace('sha(original)==NATIVE_SHA',f'_dw_hashlib.sha256(original.encode()).hexdigest()=={NATIVE_SHA!r}').replace('once(','_dw_once(')
    suffix='\nimport hashlib as _dw_hashlib\n'+function_source(parent,'once').replace('def once(','def _dw_once(')+'\nADDRESS='+repr(ADDRESS)+'\n'+fn+'\n_DW_CUDA=_dw_transform(MOE_CUDA_SRC)\n'+APPENDIX
    candidate=base+suffix;ast.parse(candidate);assert len(candidate.encode())<1_000_000
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('candidate.py',candidate),('selected.cu',native),('native.diff',''.join(difflib.unified_diff(original.splitlines(True),native.splitlines(True),fromfile='ba9-c7',tofile='direct-global-weights')))]:
        (out/name).write_text(value,encoding='utf8',newline='\n')
    m=dict(base_sha256=BASE_SHA,control_native_sha256=NATIVE_SHA,native_sha256=sha(native),candidate_sha256=sha(candidate),candidate_bytes=len(candidate.encode()),gpu=False,dispatch=False)
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    return candidate,native,m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(generate(a.out)[2],indent=2))
