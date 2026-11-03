"""Offline necessary resource/score bounds for ONE expert->slice-unpool PDL edge."""
import ast,hashlib,json,math
from pathlib import Path
PIN='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
def run(root):
    p=root/'muse_decode_layer/cand/g1b4_stage_release.py';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==PIN
    s=raw.decode();t=ast.parse(s);assign={ast.unparse(n.targets[0]):n for n in t.body if isinstance(n,ast.Assign)}
    helper=ast.literal_eval(assign["_MA_ALT_SRCS['moe_helper0']"].value)
    assert helper.count('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");')==1
    trig=helper.index('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");')
    assert helper.rfind('WS_JOIN();',0,trig)>=0 and helper.find('int* bc =',trig)>=0
    cfgs=ast.literal_eval(assign['bfp_SK2_CONFIGS'].value);cfg=next(c for c in cfgs if c[0]=='slice8_r192')
    name,layer,nrows,R,NT,NWR,WSK,UPW,NACC,CL=cfg;assert cfg==('slice8_r192','unpooler',1024,8,1,1,8,2,2,1)
    UPS=WSK*UPW;XBY=UPS*8*128;WBY=NWR*WSK*UPW*NT*768;SST=(XBY+WBY+1023)//1024*1024;NSTAGE=4
    secondary_min=1024+NSTAGE*SST+16*NSTAGE+32+4*((16*(R+1)+3)//4*4)*4+128
    native_min=3*(32768+4096)+256*4+1*8*32*4*4
    assert native_min+secondary_min>228*1024
    receipt=json.loads((root/'muse_decode_layer/profiles/scored.607eb52bdaf24160.json').read_text(encoding='utf8'))
    rows={(x['case']['load'],x['case']['layer']):x for x in receipt['result']['records']if x.get('event')=='benchmark'}
    profiles=[]
    for record in receipt['result']['records']:
        if record.get('event')=='profile' and record['case']['load'] in ('decode 1','verify 1'):
            line=[l for l in record['report'].splitlines() if 'sk2_kernel<1, 8, 1, 1, 8, 2, 4,' in l]
            assert len(line)==1 and line[0].split()[-1]=='1'
            profiles.append(dict(load=record['case']['load'],context=record['case']['layer'],line=line[0]))
    assert len(profiles)==4
    weighted={l:sum((46 if c=='sliding' else 16)*rows[l,c]['median_ns']/1000 for c in ('sliding','full'))/62 for l in ('decode 1','verify 1')}
    def factor(delta):return math.exp(sum(math.log(v/(v-delta))for v in weighted.values())/6)
    lo,hi=0.,min(weighted.values())*.99
    for _ in range(80):
        mid=(lo+hi)/2
        if factor(mid)<1.01:lo=mid
        else:hi=mid
    return dict(source_sha256=PIN,job=receipt['id'],edge='Tiny/Tiny2 expert primary -> single slice8 unpool secondary',primary_trigger_unconditional=True,producer_lower_smem=native_min,secondary_min_smem=secondary_min,secondary_pmax_extra='8*pmax',SM90_smem_capacity=228*1024,same_SM_co_residency_possible=False,consumer_grid=1024//R,profile_single_call_evidence=profiles,first_weight_bytes_all_CTAs=1024//R*WBY,stage_count=NSTAGE,initial_stage_count=1,weighted_short_us=weighted,whole_six_scenarios={str(d):factor(d)for d in (.5,1,2,3)},same_delta_per_short_context_us_for_1percent=hi,interpretation='Necessary resource/counterfactual arithmetic only; legal overlap limited to first-to-last primary retirement spread, unmeasured. No speedup or physical-bandwidth claim.',GPU=False)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(run(a.root),indent=2)+'\n',encoding='utf8')
