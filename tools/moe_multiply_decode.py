"""Owned synthesized bit permutation / multiply decoder. No donor code or table.

Mask/multiplier supplied by the coordinator's independent exhaustive search.
The table is derived from our register/half E2M1 target positions below.
"""
import ast
import hashlib
from pathlib import Path
import sys

PIN='c7854a71ad4639ed1c0ef7ff3013a831765344bf82f27caba15f034a030e03d7' # LF
M=0x81c081c0
MASK=0x001b001b
MULT=16648


def rotl(w,n):return ((w<<n)|(w>>(32-n)))&0xffffffff


def raw_decode(w):
    return w&M,rotl(w,10)&M,rotl(w,13)&M,((w&MASK)*MULT)&M


def derive_positions():
    result=[]
    for reg in range(4):
        for half in range(2):
            row=[]
            for bit in (6,7,8,15): # mantissa, exponent low/high, sign
                target=[0]*4;target[reg]=1<<(bit+16*half)
                hits=[p for p in range(32) if raw_decode(1<<p)==tuple(target)]
                assert len(hits)==1,(reg,half,bit,hits)
                row.append(hits[0])
            result.append(tuple(row))
    assert sorted(sum((list(row) for row in result),[]))==list(range(32))
    return tuple(result)


def prove_carry_freedom():
    bits=[p for p in range(32) if MASK>>p&1]
    assert len(bits)==8
    for subset in range(256):
        word=0;expected=0
        for i,p in enumerate(bits):
            if subset>>i&1:word|=1<<p;expected|=raw_decode(1<<p)[3]
        assert raw_decode(word)[3]==expected,subset


OLD='''    const unsigned r0 = w & kM;
    const unsigned r1 = (w << 6) & kM;
    const unsigned r2 = __funnelshift_r(w, w, 6) & kM;
    const unsigned r3 = ((w >> 4) & 0x00400040u) | ((w << 4) & 0x81808180u);'''
NEW='''    const unsigned r0 = w & kM;
    const unsigned r1 = __funnelshift_l(w, w, 10) & kM;
    const unsigned r2 = __funnelshift_l(w, w, 13) & kM;
    const unsigned r3 = ((w & 0x001b001bu) * 16648u) & kM;'''
OLD_NS='''    a[0] = w & kM;
    a[1] = (w << 6) & kM;
    a[2] = __funnelshift_r(w, w, 6) & kM;
    a[3] = ((w >> 4) & 0x00400040u) | ((w << 4) & 0x81808180u);'''
NEW_NS='''    a[0] = w & kM;
    a[1] = __funnelshift_l(w, w, 10) & kM;
    a[2] = __funnelshift_l(w, w, 13) & kM;
    a[3] = ((w & 0x001b001bu) * 16648u) & kM;'''


def patch(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN,'Unreviewed owned source'
    prove_carry_freedom()
    assert source.count(OLD)==3 and source.count(OLD_NS)==1
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='W4_BITPOS')
    lines=source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=['W4_BITPOS = '+repr(derive_positions())+'\n']
    out=''.join(lines).replace(OLD,NEW).replace(OLD_NS,NEW_NS)
    ast.parse(out)
    return out


if __name__=='__main__':
    src,dst=map(Path,sys.argv[1:])
    out=patch(src.read_text(encoding='utf-8'))
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(out,encoding='utf-8',newline='\n')
    print('OWNED_MULTIPLY',derive_positions(),len(out.encode()),hashlib.sha256(out.encode()).hexdigest())
