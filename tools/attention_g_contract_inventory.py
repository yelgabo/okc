"""Offline source/owning-profile census for G0->607 attention contract audit."""
import ast,hashlib,json,re
from pathlib import Path

def digest(raw):return hashlib.sha256(raw).hexdigest()
def inventory(root):
    paths={'g0':root/'.build/today/kel/kernels/muse_decode_layer.py','607':root/'muse_decode_layer/cand/g1b4_stage_release.py'}
    pins={'g0':'1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a','607':'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'}
    texts={};literals={};out={}
    for tag,p in paths.items():
        raw=p.read_bytes();assert digest(raw)==pins[tag]
        s=raw.decode();texts[tag]=s;t=ast.parse(s);literals[tag]={}
        for n in t.body:
            if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('ATT7_CUDA_SRC','ATT7_CPP_SRC'):
                v=ast.literal_eval(n.value);literals[tag][n.targets[0].id]=v
        out[tag]={'path':str(p.relative_to(root)),'sha256':digest(raw),'bytes':len(raw),'literals':{k:dict(bytes=len(v.encode()),sha256=digest(v.encode())) for k,v in literals[tag].items()}}
    g=literals['g0']['ATT7_CUDA_SRC'];c=literals['607']['ATT7_CUDA_SRC']
    reverted=c.replace('return __bfloat162float(__float2bfloat16_rn(x));','return x;').replace('programmaticStreamSerializationAllowed = 0;','programmaticStreamSerializationAllowed = 1;')
    assert reverted==g and literals['g0']['ATT7_CPP_SRC']==literals['607']['ATT7_CPP_SRC']
    assert 'cudaLaunchCooperativeKernel' not in c and 'cudaLaunchAttributeCooperative' not in c
    assert '#define OKC_SPINDBG 0' in c and '#define OKC_COOPGUARD 1' in c and '__trap()' in c
    assert 'PBULK' not in c
    out['attention_delta']={'cuda_exactly':['pr8 identity->BF16 RNE/widen','PDL host attribute1->0'],'cpp_identical':True,'PBULK_symbol_in_ATT7':False,'cooperative_launch_API_in_ATT7':False}
    receipt_path=root/'muse_decode_layer/profiles/scored.607eb52bdaf24160.json';r=json.loads(receipt_path.read_text(encoding='utf8'))
    assert r['id']=='607eb52bdaf24160' and r['board']=='muse_decode_layer' and r['status']=='done' and r['result']['passed']
    rows=[]
    for record in r['result']['records']:
        if record.get('event')!='profile':continue
        matches=re.findall(r'attn_kernel<(\d+), (true|false), (\d+)>',record['report']);assert len(matches)==1
        l,flat,mode=matches[0];l=int(l);mode=int(mode);flat=flat=='true'
        join='selected per-request software join' if not flat and mode&3==3 else 'runtime flat q.coop not exported' if flat and l==8 else 'no selected inter-CTA waiting finalize'
        rows.append(dict(load=record['case']['load'],context=record['case']['layer'],L=l,flat=flat,mode=mode,progress_scope=join))
    assert len(rows)==12 and sum(x['progress_scope']=='selected per-request software join' for x in rows)==5
    out['owning_receipt']={'job':r['id'],'sha256':digest(receipt_path.read_bytes()),'rows':rows}
    out['status']='source/profile census only; no GPU, no cost attribution'
    return out
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(inventory(a.root),indent=2)+'\n',encoding='utf8')
