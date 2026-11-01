"""Audit the timing envelope in a terminal off-board WGS frontier receipt."""
import argparse
import ast
import base64
import json
import math
import struct
import zlib
from pathlib import Path


def decode(capsule):
    raw = zlib.decompress(base64.b64decode(capsule,validate=True))
    tag,g,m = struct.unpack_from('<4sHH',raw)
    if tag != b'WGF1' or not 1 <= g <= 136 or m != 256:
        raise ValueError('unknown representative trace')
    n = 32 + 3*g + 3*m
    values = struct.unpack(f'<{n}I',raw[8:])
    writes = values[:32]
    phase_end = values[32+g:32+2*g]
    released = values[32+2*g:32+3*g]
    starts = values[32+3*g:32+3*g+m]
    ends = values[32+3*g+m:32+3*g+2*m]
    masks = values[-m:]
    used = {e for mask in masks for e in range(32) if mask & (1<<e)}
    if used != {e for e,tick in enumerate(writes) if tick}:
        raise ValueError('route coverage differs from stored writer samples')
    predicted = list(phase_end)
    for t,mask in enumerate(masks):
        dep = max((writes[e]-1 for e in used if mask & (1<<e)),default=0)
        owner = t%g
        predicted[owner] = max(dep,predicted[owner]) + ends[t]-starts[t]
    return (max(max(ends),max(released))-max(predicted))/1000


def summarize(receipt):
    if receipt['board'] != 'muse_moe_w4a16' or receipt['status'] != 'done' or not receipt['result']['passed']:
        raise ValueError('requires a passing terminal standalone MoE receipt')
    lines = receipt['result']['log_tail'].splitlines()
    if not any(l.startswith('WGSFRONT_DONE 6 ') for l in lines):
        raise ValueError('missing terminal six-case evidence')
    rows = [ast.literal_eval(l[len('WGSFRONT_CASE '):]) for l in lines if l.startswith('WGSFRONT_CASE ')]
    required={(m,s) for m in (128,256) for s in (4511,4517,4523)}
    if len(rows)!=6 or {(m,s) for m,s,*_ in rows}!=required:
        raise ValueError('incomplete coverage')
    (resline,) = [l for l in lines if l.startswith('WGSFRONT_RESOURCE ')]
    values,flag = resline[len('WGSFRONT_RESOURCE '):].split(' same_residency_no_spill ')
    resources = ast.literal_eval(values)
    resource_pass = flag == 'True' and all(len(r)==12 and r[7]==0 and r[10]>=1 for r in resources)
    no_distortion=resource_pass
    cases={}
    for m,s,med,info in rows:
        ctrl=(med[0]*med[1])**.5
        noise=abs(math.log(med[0]/med[1]))
        displacement=abs(med[2]/ctrl-1)
        no_distortion &= noise <= math.log(1.02) and displacement <= .05
        sample = info['saving_samples_us']
        if len(sample)!=5 or any(not 0 <= v < ctrl for v in sample):
            raise ValueError('unusable critical path sample')
        cases[m,s]={'control_us':ctrl,'null_log':noise,'stamp_delta':displacement,
                    'min_us':min(sample),'median_us':info['ideal_saving_us'],'max_us':max(sample)}
    (cap,)=[l.partition(' ')[2] for l in lines if l.startswith('WGSFRONT_TRACE ')]
    if not math.isclose(decode(cap),cases[256,4517]['median_us'],abs_tol=1e-9):
        raise ValueError('stored trace and computed endpoint disagree')
    bounds={}
    for s in (4511,4517,4523):
        bounds[s]={}
        for kind in ('min_us','median_us','max_us'):
            # Four unaffected official loads have factor 1. The trial never
            # treats one microsecond on each of these workloads as equal.
            x=[cases[m,s] for m in (128,256)]
            bounds[s][kind]=math.prod(r['control_us']/(r['control_us']-r[kind]) for r in x)**(1/6)
    return {'job':receipt['id'],'resources':resources,'distortion_gate':no_distortion,
            'cases':{f'{m}/{s}':v for (m,s),v in cases.items()},'schedule_factors':bounds}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(json.loads(args.receipt.read_text('utf-8-sig'))),indent=2))
