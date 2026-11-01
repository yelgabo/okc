"""Owned fixed32-unit same-expert band transpose. Source/compile research only.

The normal source stays unchanged. No decoder/packing/precision/fence is removed.
Credits publish once at each physical job's last local occurrence, not each band.
"""
import ast
import hashlib
from moe_multiply_decode import PIN

BAND=32
DEVICE=r'''
// A virtual contiguous CTA range is transposed only inside an expert's fixed8-row jobs.
// Metadata is published before issue_unit; no position_pass output is needed here.
__device__ __forceinline__ int band_map(const Smem& S, int v, bool phb) {
    const int shift = phb ? 8 : 9;
    const int j = v >> shift;
    const int e = S.job_e[j];
    const int ji = (S.job_q[j] - S.off[e]) >> 3;
    const int base = j - ji;
    const int jobs = (S.cnt[e] + 7) >> 3;
    const int x = v - (base << shift);
    const int band = x / (jobs * 32);
    const int pj = base + (x / 32) % jobs;
    const int rb = band * 32 + (x & 31);
    return (pj << shift) + rb;
}

// Number of units for physical group-job i preceding virtual local index x.
__device__ __forceinline__ int band_prefix(int x, int jobs, int i, int units) {
    x = max(0, min(x, jobs * units));
    return (x / (jobs * 32)) * 32 + min(32, max(0, x % (jobs * 32) - i * 32));
}

// Last virtual occurrence publishes this CTA's complete contribution, even across bands.
__device__ __forceinline__ int band_credit(const Smem& S, int j, int v) {
    const int e = S.job_e[j];
    const int i = (S.job_q[j] - S.off[e]) >> 3;
    const int base = j - i;
    const int jobs = (S.cnt[e] + 7) >> 3;
    const int origin = base * 512;
    const int lo = band_prefix(S.a0 - origin, jobs, i, 512);
    const int hi = band_prefix(S.a0 + S.na - origin, jobs, i, 512);
    if (hi == lo) return 0;
    const int k = hi - 1;
    const int last = origin + (k / 32) * jobs * 32 + i * 32 + (k & 31);
    return v == last ? hi - lo : 0;
}

'''

def patch(source, *, cooperative=False):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN, 'unreviewed owned source'
    if cooperative:
        from moe_owned_cooperative_pair import cooperative_source
        source=cooperative_source(source)
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SKINNY_BLOCK')
    old=node.value.value;b=old
    def once(a,z):
        nonlocal b
        assert b.count(a)==1,(a,b.count(a))
        b=b.replace(a,z)
    once('// Unit q of this CTA\'s list',DEVICE+'// Unit q of this CTA\'s list')
    once('const int v = S.a0 + q;','const int v = band_map(S, S.a0 + q, false);')
    once('const int v = S.b0 + (q - S.na);','const int v = band_map(S, S.b0 + (q - S.na), true);')
    a='const int v = phb ? b0 + (q - na) : a0 + q;'
    assert b.count(a)==2
    b=b.replace(a,'const int virtual_v = phb ? b0 + (q - na) : a0 + q;\n        const int v = band_map(S, virtual_v, phb);')
    once('    int jcred = -1;\n    int ncred = 0;\n','')
    start=b.index('        // Phase A\'s last credits go out before waiting on any phase B partials:')
    end=b.index('        const long long cw',start)
    b=b[:start]+'''        // Every local A contribution was credited at its final virtual occurrence.
        // No pending credit may cross the first phase-B partial-buffer wait.
'''+b[end:]
    start=b.index('            // Credits go out once per job:')
    end=b.index('        } else {',start)
    b=b[:start]+'''            // All stores for this physical job from this helper precede this credit.
            const int ncredit = band_credit(S, j, virtual_v);
            if (ncredit) credit(p, S, j, ncredit, lane);
'''+b[end:]
    once('    if (ncred > 0) {\n        credit(p, S, jcred, ncred, lane);\n    }\n','')
    assert 'jcred' not in b and 'ncred' not in b.replace('ncredit','')
    lines=source.splitlines(keepends=True)
    result=''.join(lines[:node.lineno-1])+ '_SKINNY_BLOCK = '+repr(b)+'\n'+''.join(lines[node.end_lineno:])
    ast.parse(result)
    return result

def group_map(x,jobs,units):
    return (x//BAND)%jobs, (x//(jobs*BAND))*BAND+x%BAND

def prefix(x,jobs,i,units):
    x=max(0,min(x,jobs*units))
    return (x//(jobs*BAND))*BAND+min(BAND,max(0,x%(jobs*BAND)-i*BAND))

def credit_at(a,b,jobs,i,units=512):
    lo=prefix(a,jobs,i,units);hi=prefix(b,jobs,i,units)
    if hi==lo:return None,0
    k=hi-1
    return (k//BAND)*jobs*BAND+i*BAND+k%BAND,hi-lo

if __name__=='__main__':
    from pathlib import Path
    import sys
    src,dst=map(Path,sys.argv[1:])
    out=patch(src.read_text(encoding='utf8'));dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(out,encoding='utf8',newline='\n')
    print('OWNED_BAND32_SOURCE',len(out.encode()),hashlib.sha256(out.encode()).hexdigest())
