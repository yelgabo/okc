"""Fixed two-WG/K128/64-token native candidate. No dispatch or default edits."""
import argparse
import ast
import difflib
import json
from pathlib import Path
from g_expert_g224_prefix import BASE, BASE_SHA, NATIVE_SHA, sha, once, literal, transform_final, function_source
from g_expert_g224_chunk_phased import NEW_BODY

ROOT=Path(__file__).resolve().parents[1]

HELPERS='''
constexpr int K128_WGS = 2, K128_MAX_GRID = 264;
__device__ __forceinline__ uint32_t k128_boundary(uint32_t total, int cta, int grid) {
  return (uint32_t)((unsigned long long)total * (unsigned)cta / (unsigned)grid);
}
'''


def transform(original):
    assert sha(original)==NATIVE_SHA
    start=original.index('#define OKC_WGSNX 3')
    end=original.index('\n#endif\n\ntemplate <bool PB, bool BIG>',start)
    r=original[start:end]
    r=once(r,'constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;', 'constexpr int WGS_PIECE = OKC_SC6W ? 4480 : 4608;')
    r=once(r,'constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;', 'constexpr int WGS_STG = 2 * WGS_PIECE;\nconstexpr int K128_RING_BYTES = 9216;'+HELPERS)
    r=once(r,'constexpr int WGS_XS = WGS_NT * 4096;', 'constexpr int WGS_XS = WGS_NT * 2048;')
    r=once(r,'constexpr int WGS_SLOTS = 160;', 'constexpr int WGS_SLOTS = 265;')
    r=r.replace('WGS_SLOTS * 4','WGS_SLOTS * K128_WGS').replace('+ 136 * 16;','+ JMAX * 32;')
    r=once(r,'MF_DA = 384, MF_INTS = 4096','MF_DA = 512, MF_INTS = 8192')
    r=r.replace('JMAX * 16','JMAX * 32').replace('j * 16 +','j * 32 +')
    r=once(r,'static_assert(BX_E + WGS_NX <= 24, "WGS barriers");', 'static_assert(BX_E + WGS_NX <= 24, "WGS barriers");\nstatic_assert(NW == 8 && NT == 256 && NTL == 320 && FT_T == 64 && WGS_NT == 8, "K128 fixed geometry");')
    r=r.replace('pb ? 4 : 5','pb ? 5 : 6').replace('PB ? 4 : 5','PB ? 5 : 6')
    # Each output unit now has 32 K stages, not 16.
    r=r.replace('gs >> 4','gs >> 5').replace('gs & 15','gs & 31')
    r=once(r,'* 16 + st)) * (size_t)WGS_STG','* 32 + st)) * (size_t)WGS_STG')
    r=r.replace('UNIT_BYTES','K128_RING_BYTES')
    r=r.replace('pd.slot * 4 + wg','pd.slot * K128_WGS + wg').replace('slot * 4 + wg','slot * K128_WGS + wg')
    r=r.replace('r16 = ub * 16 + wg * 4 + wp','r16 = ub * 8 + wg * 4 + wp')
    r=r.replace('r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp','r16 = (u & ((1 << UBL) - 1)) * 8 + wg * 4 + wp')
    # Keep two units per down K-stage, but now two WGs/unit => four releases.
    r=once(r,'mf_da(p, S) + j * 32 + st) < 8','mf_da(p, S) + j * 32 + st) < 4')
    r=r.replace('16 - pd.nst','32 - pd.nst').replace('old + pd.nst == 16','old + pd.nst == 32')
    r=r.replace('g1 & 15','g1 & 31').replace('g0 & 15','g0 & 31')
    r=r.replace('gs & 15','gs & 31').replace('gs >> 4','gs >> 5')
    r=r.replace('min(16,','min(32,').replace('min(16 -','min(32 -')
    r=r.replace('16 - s0','32 - s0').replace('s1 == 16','s1 == 32').replace('s0 == 0','s0 == 0')
    r=r.replace('s0 != 0 || s1 != 16','s0 != 0 || s1 != 32')
    r=r.replace('st = 15','st = 31')
    r=r.replace('((g1 - 1) >> 4) << 4','((g1 - 1) >> 5) << 5')
    r=r.replace('s1 - s0 == 16','s1 - s0 == 32').replace('u << 4, gridDim.x','u << 5, gridDim.x')
    # Prefix/range coordinates: doubled output units AND doubled K stages.
    r=r.replace('16u * cs','32u * cs').replace('min(15u,','min(31u,').replace('un * 16u','un * 32u')
    r=r.replace('s >> (9 - pb)','s >> (11 - pb)').replace('(1 << (9 - pb))','(1 << (11 - pb))')
    r=r.replace('(lo << (9 - pb))','(lo << (11 - pb))').replace('(nj << (9 - pb))','(nj << (11 - pb))').replace('nj << (9 - pbl)','nj << (11 - pbl)')
    r=r.replace('(sj >> 4)','(sj >> 5)').replace('(sj & 15u)','(sj & 31u)')
    r=r.replace('wgs_cost(0, S.jC[j]) * 16u','wgs_cost(0, S.jC[j]) * 32u').replace('wgs_cost(1, S.jC[j]) * 16u','wgs_cost(1, S.jC[j]) * 32u')
    r=r.replace('OKC_WGSU) << 5;','OKC_WGSU) << 6;').replace('OKC_WGSUB) << 4;','OKC_WGSUB) << 5;')
    for index in ('c','c + 1','c + kk'):
        r=r.replace('tot * (uint32_t)'+('('+index+')' if '+' in index else index)+' / (uint32_t)G',f'k128_boundary(tot, {index}, G)')
    # K128 activation layout/permutation, for both gather and fused SiLU store.
    a=r.index('__device__ void wgs_gather(');b=r.index('// Work left from the previous segment',a)
    gather=NEW_BODY.replace('G224','K128')
    gather=gather.replace('column >> 5','column >> 4').replace('column & 31','column & 15').replace(' * 32 + (column & 15)',' * 16 + (column & 15)')
    gather=gather.replace('const int group = column & 15;','const int group = column & 15;').replace(' * 256)',' * 128)')
    gather=gather.replace('8 * ((group >> 1) & 3) + 2 * (group >> 3)', '8 * ((group >> 1) & 1) + 2 * (group >> 2)')
    gather=gather.replace('xs32 + 32','xs32 + 16').replace('xs32 + 64','xs32 + 32').replace('xs32 + 96','xs32 + 48')
    r=r[:a]+gather+'\n'+r[b:]
    r=r.replace('npad * 512u','npad * 256u').replace('st * npad * 256','st * npad * 128')
    r=r.replace('col >> 8','col >> 7').replace('col & 255','col & 127').replace('* 32 + ((col & 127)','* 16 + ((col & 127)')
    # K128 has 4 load groups with 2 K16 fragments each: preserve group16 scales.
    a=r.index('    uint4 cwn =');b=r.index('        wg_fence();',a)
    old=r[a:b]
    new='''    uint2 cwn = *reinterpret_cast<const uint2*>(pc + (wp * 32 + lane) * 8);
    uint32_t scw;
#if OKC_SC6W
    const int tt = wp * 32 + lane;
    const uint32_t p0 = pc[4096 + tt], p1 = pc[4096 + 128 + tt], p2 = pc[4096 + 256 + tt];
    const uint32_t packed = p0 | (p1 << 8) | (p2 << 16);
    scw = ((packed & 63u) + 64u) | ((((packed >> 6) & 63u) + 64u) << 8)
        | ((((packed >> 12) & 63u) + 64u) << 16) | ((((packed >> 18) & 63u) + 64u) << 24);
#else
    scw = *reinterpret_cast<const uint32_t*>(pc + 4096 + (wp * 32 + lane) * 4);
#endif
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint2 cwg = cwn;
      if (lg < 3) cwn = *reinterpret_cast<const uint2*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 8);
      else {
        asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
        mbar_arrive1(S.bars + BW_E + ws);
      }
#pragma unroll
      for (int h = 0; h < 2 / GS; ++h) {
        const int ci = lg * (2 / GS) + h;
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;
          const uint32_t q = js == 0 ? cwg.x : cwg.y;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          const uint32_t sb = (uint32_t)js * 2u;
          const uint32_t sgp = __byte_perm(scw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
        }
'''
    r=r[:a]+new+r[b:]
    r=r.replace('(lg * 4 + h * GS + jj)','(lg * 2 + h * GS + jj)')
    r=once(r,'wg_desc(xring_s, 128, 4096)','wg_desc(xring_s, 128, 2048)')
    r=once(r,'(OKC_SC6W ? 4u : 16u) + p.zero','16u + p.zero')
    # K128 has four GS2 commit groups (ci0..3). Original FCI4 would
    # never flush a pending segment before pd is overwritten.
    r=once(r,'FIRST && ci == OKC_WGSFCI && pd.kind','FIRST && ci == 3 && pd.kind')
    changed=original[:start]+r+original[end:]
    changed=once(changed,'__launch_bounds__(NTL, 1) moe_fused','__launch_bounds__(NTL, 2) moe_fused')
    changed=once(changed,'okc::WGS_NS * okc::UNIT_BYTES','okc::WGS_NS * okc::K128_RING_BYTES')
    changed=changed.replace('okc::WGS_SLOTS * 4','okc::WGS_SLOTS * okc::K128_WGS')
    changed=changed.replace('JMAX * 16','JMAX * 32')
    changed=changed.replace('i < JMAX * 16; i += G * NT) p.doneAs','i < JMAX * 32; i += G * NT) p.doneAs')
    changed=once(changed,'grid>0 && grid<=prop.multiProcessorCount && grid<=136 && grid<160','(grid *= 2)>0 && grid==2*prop.multiProcessorCount && grid<=okc::K128_MAX_GRID && grid<okc::WGS_SLOTS')
    changed=once(changed,'resident>=1 && grid<=resident*prop.multiProcessorCount','resident>=2 && grid<=resident*prop.multiProcessorCount')
    changed=once(changed,'#include <torch/extension.h>','#include <torch/extension.h>\n#include <cstdio>')
    changed=once(changed,'  void* args[]={&p};', '''  static unsigned admission_seen = 0;
  const unsigned admission_bit = p.M>=OKC_BIGM ? 2u : 1u;
  if (!(admission_seen & admission_bit)) {
    cudaFuncAttributes actual{};
    TORCH_CHECK(cudaFuncGetAttributes(&actual,kernel)==cudaSuccess,"K128_RESOURCE attributes");
    std::printf("K128_ADMISSION b4=%d big=%d m=%d sm=%d grid=%d threads=%d dynamic=%zu static=%zu local=%zu regs=%d resident=%d\\n",
      OKC_B4FIN,int(p.M>=OKC_BIGM),p.M,prop.multiProcessorCount,grid,okc::NTL,
      sm,actual.sharedSizeBytes,actual.localSizeBytes,actual.numRegs,resident);
    std::fflush(stdout);
    admission_seen |= admission_bit;
  }
  void* args[]={&p};''')
    changed=transform_final(changed).replace('G224','K128')
    return changed


PACKERS='''
def _kperm_src():
    p = torch.arange(128)
    s, c = p >> 4, p & 15
    lg, j, t, i = s >> 1, s & 1, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
    k = 16 * (2 * t + j) + 4 * lg + i
    assert sorted(k.tolist()) == list(range(128))
    return k

def _kperm_codes(codes):
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 128, 128)
    nib = nib.index_select(2, _kperm_src().to(codes.device)).reshape(R, K // 2, 2)
    return nib[..., 0] | (nib[..., 1] << 4)

def _kperm_rows(rowsA):
    r = rowsA.view(-1, 2, 8)
    gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
    P = torch.arange(gate.numel(), device=rowsA.device)
    cm = (P >> 7) * 128 + _kperm_src().to(rowsA.device)[P & 127]
    return torch.stack((gate[cm].view(-1, 8), up[cm].view(-1, 8)), 1).reshape(-1)

def _wgs_copy(codes, sf, rows, sc6=False):
    wn = _pack_codes_nat(_kperm_codes(codes), rows)
    nb, nub = rows.numel() // 16, rows.numel() // 128
    w = wn.view(nb, 8, 8, 8, 4, 4).permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)
    w = w.view(nub, 2, 4, 32, 4, 2, 32).permute(0, 3, 1, 4, 2, 6, 5).contiguous()
    cb = w.view(torch.uint8).view(nub, 32, 2, 4096)
    s = sf.view(torch.uint8).index_select(0, rows)
    s = s.view(nub, 2, 4, 2, 8, 32, 4, 2).permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 32, 2, 128, 4)
    if sc6:
        v = s.to(torch.int32) - 64
        assert int(v.min()) >= 0 and int(v.max()) <= 63
        p = v[..., 0] | (v[..., 1] << 6) | (v[..., 2] << 12) | (v[..., 3] << 18)
        s = torch.stack((p & 255, (p >> 8) & 255, (p >> 16) & 255), -2).to(torch.uint8).reshape(nub, 32, 2, 384)
    else:
        s = s.reshape(nub, 32, 2, 512)
    return torch.cat((cb, s), dim=3).contiguous().view(-1)
'''

APPENDIX='''
_K128_PARENT_EXT = _ma_ext
_K128_EXTS = {}
def _ma_ext(defines=None):
    defines = tuple(MOE_DEFINES if defines is None else defines)
    if "-DOKC_WGS=1" not in defines:
        return _K128_PARENT_EXT(defines)
    if any(v.startswith("@") for v in defines):
        raise RuntimeError("K128 unallocated alternate WGS source")
    if DQ7 and "-DOKC_DQ7=1" not in defines: defines += ("-DOKC_DQ7=1",)
    if defines not in _K128_EXTS:
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) != (9,0): raise RuntimeError("K128 requires SM90")
        defs = ["-DOKC_NW=8" if d=="-DOKC_NW=16" else d for d in defines]
        import hashlib
        tag = hashlib.sha256((_K128_CUDA+repr(defs)).encode()).hexdigest()[:16]
        _K128_EXTS[defines] = load_inline(name="wgs_k128_"+tag, cpp_sources=MOE_CPP_SRC, cuda_sources=_K128_CUDA,
            functions=["okc_run","okc_set_pdl","okc_attrs"], extra_cflags=["-std=c++20"],
            extra_cuda_cflags=["-O3","-std=c++20","-gencode=arch=compute_90a,code=sm_90a","-lineinfo", "-DOKC_NS=3","-DOKC_FNST=4","-DOKC_NSF=3",*defs],verbose=False)
        _K128_EXTS[defines].okc_set_pdl(0)
    return _K128_EXTS[defines]

def _ma_make_scratch(dev):
    xrows = _MA_EMAX + 8 * (32 + _MA_EMAX // 64 + 8)
    fixup = 2 * 265 * 2 * 4096 + 2 * 265 * 2 + (32 + _MA_EMAX // 64 + 8) * 32
    return [torch.zeros(8 + 1024 + _MA_EMAX + 8192,dtype=torch.int32,device=dev),
            torch.zeros(_MA_EMAX + _MA_EMAX * 4096 + fixup,dtype=torch.float32,device=dev),
            _df_inter(torch.zeros((_MA_EMAX+2*xrows)*4096,dtype=torch.bfloat16,device=dev))]

def _fork_install_fallback(weights, state, reason):
    raise RuntimeError("K128_STOP no silent fallback: "+reason)
'''


def generate(out):
    base=(ROOT/BASE).read_text();assert sha(base)==BASE_SHA
    original=literal(base,'MOE_CUDA_SRC');native=transform(original)
    a,b=original.splitlines(True),native.splitlines(True)
    own=Path(__file__).read_text()
    parent=(ROOT/'tools/g_expert_g224_prefix.py').read_text()
    fn=function_source(own,'transform').replace('def transform(', 'def _k128_transform(')
    fn=fn.replace('sha(original)==NATIVE_SHA',f'_k128_hashlib.sha256(original.encode()).hexdigest()=={NATIVE_SHA!r}')
    fn=fn.replace('once(', '_k128_once(').replace('transform_final(', '_k128_finalize(').replace('HELPERS','_K128_HELPERS').replace('NEW_BODY','_K128_GATHER')
    final=function_source(parent,'transform_final').replace('def transform_final(', 'def _k128_finalize(').replace('once(', '_k128_once(')
    suffix='\nimport hashlib as _k128_hashlib\n'+function_source(parent,'once').replace('def once(', 'def _k128_once(')+'\n'
    suffix+=f'_K128_HELPERS={HELPERS!r}\n_K128_GATHER={NEW_BODY!r}\n'+final+'\n'+fn+'\n_K128_CUDA = _k128_transform(MOE_CUDA_SRC)\n'
    suffix+=PACKERS+APPENDIX
    candidate=base+suffix
    assert len(candidate.encode())<1_000_000,len(candidate.encode())
    ast.parse(candidate)
    out.mkdir(parents=True,exist_ok=True)
    for name,s in [('candidate.py',candidate),('selected.cu',native),('native.diff',''.join(difflib.unified_diff(a,b)))]:
        (out/name).write_text(s,encoding='utf8',newline='\n')
    manifest=dict(base_sha256=BASE_SHA,native_sha256=sha(native),candidate_sha256=sha(candidate),candidate_bytes=len(candidate.encode()),gpu=False,dispatch=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return candidate,native,manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(generate(a.out)[2],indent=2))
