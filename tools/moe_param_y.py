"""Combine the baseline's prepared gamma contract with exact factorized output storage."""
import argparse
import json
from pathlib import Path
import struct

from cuda_issue_groups import generate,compile_pair
from moe_factor_y import transform as factorized,HELPER as FACTOR_HELPER

OLD='return make_float2(__fmul_rn(down.x,fork_bf16(1.f + raw.x)),__fmul_rn(down.y,fork_bf16(1.f + raw.y)));'
NEW='return make_float2(__fmul_rn(down.x,raw.x),__fmul_rn(down.y,raw.y));'
HELPER=FACTOR_HELPER.replace(OLD,NEW)


def gamma(raw):
    value=struct.unpack('<f',struct.pack('<I',raw<<16))[0]
    encoded=struct.unpack('<I',struct.pack('<f',1.+value))[0]
    return ((encoded+0x7fff+((encoded>>16)&1))>>16)&0xffff


def reconstruct(down,prepared):
    a=struct.unpack('<f',struct.pack('<I',down<<16))[0]
    b=struct.unpack('<f',struct.pack('<I',prepared<<16))[0]
    try:return struct.unpack('<I',struct.pack('<f',a*b))[0]
    except OverflowError:return 0xff800000 if a*b<0 else 0x7f800000


def transform(native):
    source,edits=factorized(native)
    old='const float w0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0))), w1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0 + 8)));'
    new='const float w0 = __bfloat162float(__ldg(p.normw + n0)), w1 = __bfloat162float(__ldg(p.normw + n0 + 8));'
    changes=[(OLD,NEW,1),(old,new,1)]
    for before,after,count in changes:
        assert source.count(before)==count
        source=source.replace(before,after)
    return source,edits+changes


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--compile',action='store_true');args=parser.parse_args()
    print(json.dumps(generate(args.out,transform),indent=2),flush=True)
    if args.compile:compile_pair(args.out)
