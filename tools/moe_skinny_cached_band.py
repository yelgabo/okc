"""Same band32 ordering/credits, precomputed into bounded packed shared schedule.

Research-only; no production source change or GPU authorization. H200 grid>=132,
M<=32 are mandatory host constraints, not empirical capacity assumptions.
"""
import ast
import hashlib
from moe_skinny_band import patch as band_patch

ENTRIES=350

def patch(source, *, cooperative=False):
    source=band_patch(source,cooperative=cooperative)
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SKINNY_BLOCK')
    b=node.value.value
    def once(old,new):
        nonlocal b
        assert b.count(old)==1,(old,b.count(old))
        b=b.replace(old,new)
    once('    int njobs;\n', '    unsigned schedule[350]; // low16 physical v; high16 final-occurrence A credit\n    int njobs;\n')
    once('band_map(S, S.a0 + q, false)', '(S.schedule[q] & 65535u)')
    once('band_map(S, S.b0 + (q - S.na), true)', '(S.schedule[q] & 65535u)')
    assert b.count('band_map(S, virtual_v, phb)')==2
    b=b.replace('band_map(S, virtual_v, phb)', '(S.schedule[q] & 65535u)')
    once('const int ncredit = band_credit(S, j, virtual_v);','const int ncredit = S.schedule[q] >> 16;')
    old='''            const int n = (a1 - a0) + (b1 - b0);
            for (int q = 0; q < min(kNS, n); ++q) {
                issue_unit(p, S, q, pol);
            }
        }
    } else {'''
    new='''        }
        // All ranges and immutable job metadata are helper-warp stores.
        __syncwarp();
        const int n = S.na + S.nb;
        // Host contract M<=32, G>=132 proves n<=350 and v<30720.
        for (int q = lane; q < n; q += 32) {
            const bool phb = q >= S.na;
            const int vv = phb ? S.b0 + q - S.na : S.a0 + q;
            const int v = band_map(S, vv, phb);
            const int nc = phb ? 0 : band_credit(S, v >> 9, vv);
            S.schedule[q] = static_cast<unsigned>(v) | (static_cast<unsigned>(nc) << 16);
        }
        __syncwarp();
        if (lane == 0) {
            for (int q = 0; q < min(kNS, n); ++q) issue_unit(p, S, q, pol);
        }
    } else {'''
    once(old,new)
    signature='void launch(const Params& p, int sms, cudaStream_t stream) {'
    once(signature,signature+'\n    TORCH_CHECK(p.M<=32 && sms>=132, "OWNED_CACHED_BAND: capacity contract");')
    lines=source.splitlines(keepends=True)
    result=''.join(lines[:node.lineno-1])+'_SKINNY_BLOCK = '+repr(b)+'\n'+''.join(lines[node.end_lineno:])
    ast.parse(result)
    return result

if __name__=='__main__':
    from pathlib import Path
    import sys
    src,dst=map(Path,sys.argv[1:])
    out=patch(src.read_text(encoding='utf8'));dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(out,encoding='utf8',newline='\n')
    print('OWNED_CACHED_BAND_SOURCE',len(out.encode()),hashlib.sha256(out.encode()).hexdigest())
