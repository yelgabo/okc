"""Source-bound CPU validation for a future owner-produced Tiny timeline."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import zlib

from d32_b4_prefix_audit import body
from expert_top100_pregamma import capture
from gopt_ws_owned import RELEASE_PIN, HELPER_PIN, ROOT
from prod_size_compile_g import bindings



def require(condition, message):
    if not condition:
        raise ValueError('TINY_TRACE: ' + message)


def source_map():
    source = (ROOT/'muse_moe_w4a16/cand/current_control_release.py').read_text()
    require(hashlib.sha256(source.encode()).hexdigest() == RELEASE_PIN, 'release source pin')
    helper = bindings(source)['_MA_ALT_SRCS']['moe_helper0']
    require(hashlib.sha256(helper.encode()).hexdigest() == HELPER_PIN, 'selected helper pin')
    calls, routes = capture(source)
    sections = dict(kernel=body(helper, '__global__ void __launch_bounds__(NTL, 1) moe_fused('),
                    consumers=body(helper, '__device__ void skinny_ring_ws('),
                    helper=body(helper, '__device__ void ws_helper('))
    signature='__device__ __forceinline__ void ws_issue('
    sections['issue']=body(helper[helper.rfind(signature):],signature)
    anchors = {
        'kernel_entry': ('kernel', '  const int tid = TIDX;'),
        'fast_prefill_issue': ('kernel', '        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;'),
        'later_refill_issue': ('issue', '  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);'),
        'ring_ready': ('consumers', '    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));'),
        'consumer_partial_join': ('consumers', '    CTA_SYNC();'),
        'phase_b_job_wait': ('consumers', '        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }'),
        'compute_tail': ('consumers', '  if (PB && warp < OKC_WSBLK && warp < nU) {'),
        'helper_count': ('helper', '  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;'),
        'a_credit': ('helper', '            atomicAdd(p.doneA + jj, b - a);'),
        'helper_fenced_exit': ('helper', '  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier'),
        'final_join': ('kernel', "  WS_JOIN();  // the helper's last epilogue stores are visible"),
        'grid_wait': ('kernel', '    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);'),
        'finish': ('kernel', '  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered'),
    }
    result = {}
    for name, (section, needle) in anchors.items():
        text = sections[section]
        require(text.count(needle) == 1, 'native anchor drift: ' + name)
        offset = helper.index(text) + text.index(needle)
        result[name] = dict(line=helper[:offset].count('\n')+1, anchor=needle.strip())
    builds = {}
    for tokens, call, route in zip((1,8), calls[:2], routes[:2]):
        require(route['tokens'] == tokens, 'factory route')
        flags = call['extra_cuda_cflags']
        tail = next(int(f.split('=')[1]) for f in flags if f.startswith('-DOKC_WSBLK='))
        builds[str(tokens)] = dict(key=route['key'], control_flags=flags, tail_warps=tail)
    return dict(release_sha256=RELEASE_PIN, helper_sha256=HELPER_PIN,
                builds=builds, anchors=result, gpu=False)


def jobs_for(ids):
    require(isinstance(ids,list) and len(ids) in (1,8), 'T1/T8 only')
    counts = [0]*32
    for row in ids:
        require(isinstance(row,list) and len(row)==8 and len(set(row))==8, 'distinct top8')
        require(all(type(e) is int and 0<=e<256 for e in row), 'global expert IDs')
        for e in row:
            if e<32: counts[e]+=1
    return [(expert,count) for expert,count in enumerate(counts) if count]


def cta_plan(ids):
    jobs = jobs_for(ids)
    tail = 2 if len(ids)==1 else 1
    result = []
    for cta in range(132):
        a0,a1 = len(jobs)*512*cta//132,len(jobs)*512*(cta+1)//132
        b0,b1 = len(jobs)*256*cta//132,len(jobs)*256*(cta+1)//132
        count = a1-a0+b1-b0
        require(count<=192, 'fixed trace capacity')
        split = count-min(tail,b1-b0)
        waits = list(range(b0//256,(b1-1)//256+1)) if b1>b0 else []
        result.append(dict(cta=cta,a=[a0,a1],b=[b0,b1],units=count,helper_units=split,wait_jobs=waits))
    return result


def audit_native_buffer(tokens, routing_ids, values):
    require(tokens==len(routing_ids), 'input token count')
    require(len(values)==136*1184 and all(type(v) is int and v>=0 for v in values), 'raw native buffer')
    plan=cta_plan(routing_ids)
    for work in plan:
        row=values[work['cta']*1184:(work['cta']+1)*1184]
        require(row[17:21]==work['a']+work['b'], 'actual CTA ranges from routing')
        require(row[21]==(2 if tokens==1 else 1) and row[22]==1, 'selected WSBL tail flags')
        a_count=work['a'][1]-work['a'][0]
        helper_last=0
        retire=0
        for q in range(work['units']):
            dep0,dep1,begin,ready,end,epi=row[32+q*6:38+q*6]
            require(retire<=begin<=ready<=end, 'single consumer stream')
            require((epi>0)==(q<work['helper_units']), 'helper versus retained-tail coverage')
            if epi:
                require(end<=epi and helper_last<=epi<=row[16], 'helper stream order')
                helper_last=epi
            first_of_job=q>=a_count and (q==a_count or (work['b'][0]+q-a_count)%256==0)
            require(bool(dep0)==bool(dep1)==first_of_job, 'every B job readiness wait')
            if first_of_job:require(row[4]<=dep0<=dep1<=begin, 'readiness before input load')
            retire=end
    from moe_tiny_timeline import trace
    exported=trace(values,132)
    return audit_exported_case(tokens,exported)


def audit_exported_case(tokens, item):
    require(tokens in (1,8) and item['tick_ns']==10, 'T1/T8 quantization')
    rows=item['ctas']
    require(len(rows)==132 and [r[0] for r in rows]==list(range(132)), 'every CTA exactly once')
    require(all(len(r)==14 and all(type(v) is int and v>=0 for v in r) for r in rows), 'integer CTA summary')
    require(all(r[1:9]==sorted(r[1:9]) for r in rows), 'summary boundary order')
    a,b=sum(r[9] for r in rows),sum(r[10] for r in rows)
    require(a%512==0 and b%256==0 and a//512==b//256, 'same A/B job extent')
    jobs=a//512
    require(jobs<=min(32,tokens*8), 'short-load job bound')
    for c,r in enumerate(rows):
        require(r[9]==jobs*512*(c+1)//132-jobs*512*c//132
                and r[10]==jobs*256*(c+1)//132-jobs*256*c//132, 'per-CTA assignment coverage')
    critical=item['critical_units']
    require(len(critical)==2 and len({r[0] for r in critical})==2, 'two distinct critical CTA witnesses')
    second=sorted((r[8] for r in rows),reverse=True)[1]
    for c,units in critical:
        require(type(c) is int and 0<=c<132 and rows[c][8]>=second, 'last sampled thread0 witness')
        r=rows[c];count=r[9]+r[10];tail=min(2 if tokens==1 else 1,r[10]);nh=count-tail
        require(count<=192 and len(units)==count, 'critical unit coverage')
        base=0;retire=0;helper_last=0
        for q,event in enumerate(units):
            require(len(event)==6 and all(v is None or type(v) is int for v in event), 'unit delta schema')
            delta,wait,compute,lag,dep_offset,dep_wait=event
            require(delta>=0 and wait>=0 and compute>=0 and (lag is None or lag>=0), 'nonnegative event durations')
            base+=delta;ready=base+wait;end=ready+compute
            require(retire<=base<=ready<=end<=r[5], 'rounded consumer timeline')
            require((lag is not None)==(q<nh), 'helper and retained-tail detail')
            if lag is not None:
                require(helper_last<=end+lag, 'rounded helper stream')
                helper_last=end+lag
            first=q==r[9] or (q>r[9] and (jobs*256*c//132+q-r[9])%256==0)
            require((dep_offset is not None)==(dep_wait is not None)==first, 'retained B readiness coverage')
            if first:
                require(dep_wait>=0 and r[4]<=base+dep_offset<=base+dep_offset+dep_wait<=base, 'rounded B wait')
            retire=end
    return dict(schema_valid=True,phase_gain_qualified=False,jobs=jobs,ctas=132,units=a+b,
                retained_cta_details=[r[0] for r in critical],
                limit='Only two CTA unit streams selected by thread0 finish survive export. B-end is thread0 return, not CTA completion. No critical-path gain is established.')


def encode(report, limit=12000):
    payload=json.dumps(report,separators=(',',':'),allow_nan=False).encode()
    encoded=base64.b85encode(zlib.compress(payload,9)).decode()
    require(len(encoded)<limit, f'payload {len(encoded)} exceeds {limit}; no records may be silently dropped')
    require(json.loads(zlib.decompress(base64.b85decode(encoded)))==report, 'lossless capsule')
    return encoded


def observer_costs(rows):
    require([r[:3] for r in rows]==[[1102031,t,'normal'] for t in (1,8,16,32,128,256)], 'six fixed cost rows')
    costs=[]
    for row in rows:
        values=row[3]
        require(len(values)==4 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values), 'finite placement medians')
        control=(values[0]+values[1])/2
        observed=(values[2]+values[3])/2
        duplicate=max(abs(values[0]-values[1]),abs(values[2]-values[3]))
        limit=max(2.,.1*control);noise_limit=max(.5,.02*control)
        costs.append([row[1],control,observed,observed-control,limit,duplicate,noise_limit,
                      abs(observed-control)<=limit and duplicate<=noise_limit])
    return costs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-map',type=Path)
    p.add_argument('--trace',type=Path)
    p.add_argument('--job')
    p.add_argument('--instrument-sha256')
    args=p.parse_args()
    if args.source_map:
        args.source_map.parent.mkdir(parents=True,exist_ok=True)
        args.source_map.write_text(json.dumps(source_map(),indent=2)+'\n')
        print('Source map only; no native observer or GPU execution')
    if args.trace:
        record=json.loads(args.trace.read_text())
        require(re.fullmatch('[a-f0-9]{16}',args.job or '') is not None and re.fullmatch('[a-f0-9]{64}',args.instrument_sha256 or '') is not None, 'expected own identifiers')
        require(record['job']==args.job and record['instrument_sha256']==args.instrument_sha256, 'expected own source/job')
        print(json.dumps(audit_exported_case(record['tokens'],record['trace']),indent=2))
