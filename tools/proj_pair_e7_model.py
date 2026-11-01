"""CPU distribution proxy and exact-byte gate for pair5-e7 + two sign bits.

No torch/CUDA; samples Gaussian weights, not the judge's GPU generator.
"""
from collections import Counter
import json
import math
import random
import struct
import sys

def model(seed=71031,count=262144):
    r=random.Random(seed)
    def bf(x):
        b=struct.unpack('<I',struct.pack('<f',x))[0]
        return ((b+32767+((b>>16)&1))>>16)&65535
    values=[bf(r.gauss(0,1)/math.sqrt(8192)) for _ in range(count)]
    pairs=[((a>>8)&127)|(((b>>8)&127)<<7) for a,b in zip(values[::2],values[1::2])]
    lut=[x for x,_ in Counter(pairs).most_common(32)]
    mapping={x:i for i,x in enumerate(lut)}
    for x in set(pairs)-mapping.keys():
        mapping[x]=max(range(32),key=lambda j:int((x&127)==(lut[j]&127))+int((x>>7)==(lut[j]>>7)))
    esc=0
    codes=[]
    for i,pair in enumerate(pairs):
        a,b=values[2*i:2*i+2];idx=mapping[pair]
        code=idx|((a>>15)<<5)|((b>>15)<<6);codes.append(code)
        ee=lut[idx]
        for h,v in enumerate((a,b)):
            high=((ee>>(7*h))&127)|(((code>>(5+h))&1)<<7)
            decoded=(v&255)|(high<<8)
            if decoded!=v:esc+=1;decoded=v
            assert decoded==v
    stored=b''.join(sum(codes[i+j]<<(7*j) for j in range(8)).to_bytes(7,'little') for i in range(0,len(codes),8))
    for i,c in enumerate(codes):
        assert ((int.from_bytes(stored[i//8*7:i//8*7+7],'little')>>(i%8*7))&127)==c
    return {'seed':seed,'count':count,'escape_fraction':esc/count,
            'bits_with_8byte_escape_entry':11.5+64*esc/count,
            'unmodeled':'per-CTA table padding, decode instructions, GPU Philox differs',
            'stored_code_bytes':len(stored),'lut_entries':len(lut),'roundtrip':True}

if __name__=='__main__':
    print(json.dumps([model(s) for s in (71031,71032,71033)],indent=2))
