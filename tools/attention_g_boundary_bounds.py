"""Offline source/profile screen for disabled GLE O-boundary integration. No kernel import."""
import ast,hashlib,json,math,re
from pathlib import Path

def run(root):
    pins={'G0':(' .build/today/kel/kernels/muse_decode_layer.py'.strip(),'1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a'),'607':('muse_decode_layer/cand/g1b4_stage_release.py','ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab')}
    source={}
    for k,(path,pin)in pins.items():
        raw=(root/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
        tree=ast.parse(raw.decode());v={}
        for n in tree.body:
            if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')in ('GLE_T','GLE_B1X','FORCE_GLUEV2'):
                v[n.targets[0].id]=ast.literal_eval(n.value)
        source[k]=dict(path=path,sha256=pin,policies=v)
    receipts={'G0':root/'muse_decode_layer/profiles/kel.3a5f879df0d44825.json','607':root/'muse_decode_layer/profiles/scored.607eb52bdaf24160.json'}
    profiles={};base={};b1={}
    for k,p in receipts.items():
        r=json.loads(p.read_text(encoding='utf8'));profiles[k]=[]
        for row in r['result']['records']:
            if row.get('event')=='benchmark'and k=='607':base[row['case']['load'],row['case']['layer']]=row['median_ns']/1000
            if row.get('event')=='profile'and row['case']['load']in ('decode 1','verify 1'):
                lines=[l for l in row['report'].splitlines()if any(n in l for n in ('glue_cluster_boundary','_gle_route','_sl_tile_route'))]
                profiles[k].append(dict(load=row['case']['load'],context=row['case']['layer'],selected=lines))
                if k=='607':
                    line=next(l for l in lines if 'glue_cluster_boundary'in l);b1[row['case']['load'],row['case']['layer']]=float(re.search(r'\s([0-9.]+)\s+µs',line)[1])
    assert source['G0']['policies']['GLE_T']=={1:True}and source['607']['policies']['GLE_T']=={}
    def scenario(loads):
        log=0
        for l in loads:
            old=sum(w*base[l,c]for w,c in ((46,'sliding'),(16,'full')))
            new=old-sum(w*b1[l,c]for w,c in ((46,'sliding'),(16,'full')))
            log+=math.log(old/new)/6
        return math.exp(log)
    return dict(source=source,profiles=profiles,b1_us={l+' '+c:v for(l,c),v in b1.items()},counterfactual_zero_b1_D1_only=scenario(('decode 1',)),counterfactual_zero_b1_bothshort=scenario(('decode 1','verify 1')),current_boundary_output_bytes_per_token=8192*(2+4+4)+4,last_owner_extra_partial_ss_bytes_per_token_each_direction=256*4,interpretation='Profile-activity substitution screen only: no exposed gain, hard speed ceiling, new implementation or eligibility claim.',GPU=False)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(run(a.root),indent=2)+'\n',encoding='utf8')
