"""Replace short-helper A-ready aggregation with private CTA flags."""
from moe_tiny_timeline import HELPER_PIN
from cuda_issue_groups import sha

HELPERS=r'''
static_assert(JMAX <= 128 && 128 + 136 <= 512, "short completion flags overlap scratch");
__device__ __forceinline__ void short_ready_wait(const Params& p,const Smem& S,int job,int lane) {
  const int total = S.njS * RB_A, G = gridDim.x;
  const int L = (job-S.njF) * RB_A, R = L + RB_A;
  const int first = ((L+1)*G + total-1)/total - 1;
  const int last = (R*G + total-1)/total - 1;
  while (true) {
    int ready = 1;
    for (int c=first+lane;c<=last;c+=32) ready &= ld_acquire(p.doneA+128+c) != 0;
    if (__all_sync(0xffffffffu,ready)) break;
    __nanosleep(64);
  }
}
'''


def contributors(jobs,grid,job):
    total=jobs*512
    return range(((job*512+1)*grid+total-1)//total-1,
                 ((job+1)*512*grid+total-1)//total)


def transform(native):
    assert sha(native)==HELPER_PIN
    edits=[]
    def edit(old,new,scope=None):
        nonlocal native
        if scope:
            start=native.index(scope[0]);end=native.index(scope[1],start)
            part=native[start:end];assert part.count(old)==1
            native=native[:start]+part.replace(old,new)+native[end:]
        else:
            assert native.count(old)==1,old[:80]
            native=native.replace(old,new)
        edits.append((old,new,1))
    edit('template <bool PB>\n__device__ void skinny_ring_ws(',HELPERS+'\ntemplate <bool PB>\n__device__ void skinny_ring_ws(')
    edit('        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }',
         '        short_ready_wait(p,S,j,lane);',
         ('__device__ void skinny_ring_ws(','__device__ __forceinline__ void ws_issue('))
    old='''        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }'''
    edit(old,'''        if (lane == 0)
          asm volatile("st.relaxed.gpu.global.u32 [%0], %1;\\n" ::"l"(p.doneA+128+blockIdx.x),"r"(1):"memory");''')
    old='    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;'
    edit(old,'    if (tid == 0) p.doneA[128+blockIdx.x] = 0;\n'+old)
    return native,edits
