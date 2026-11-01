"""Read-only attention opportunity arithmetic and source pins; no GPU or kernel import."""
import argparse
import hashlib
import json
from pathlib import Path

def inspect(root):
    external=root/'.build/today/round2-published-rohan/muse_attention_decode.py'
    vp=root/'.build/today/kel/kernels/muse_attention_decode.py'
    own=root/'muse_attention_decode/attn_one.cu'
    launch=root/'muse_attention_decode/submission.py'
    texts={p.name+str(i):p.read_text(encoding='utf8') for i,p in enumerate((external,vp,own,launch))}
    ex,vs,os,ls=texts.values()
    def sha(s):return hashlib.sha256(s.encode()).hexdigest()
    assert sha(ex)=='394f2517ca8eaf69b7d422832edfd59451a9f73103981d13b1059070abb17b4b'
    assert hashlib.sha256(vp.read_bytes()).hexdigest()=='496d460e0bb1e7489684de2742f1ff22f3959d9d5fe30674199e74b3d002d651'
    assert '(1, 1, False, 1): dict(nc=8, coop=1, n_splits=252)' in ex
    plan=ls[ls.index('def _one_plan('):ls.index('def _one(',ls.index('def _one_plan('))]
    assert 'n = min(tiles, 252)' in plan and 'nc = 8' in plan and 'coop = True' in plan
    vp2=vs[vs.index('#define VP2_PUSH'):vs.index('#undef VP2_PUSH')]
    assert 'b0 *= inv; b1 *= inv;\n      float ss = b0 * b0 + b1 * b1;' in vp2
    assert 'st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32' in vp2
    assert 'barrier.cluster.wait.acquire.aligned' in vp2
    receipt=json.loads((root/'muse_attention_decode/profiles/wait-trace.3f44f891d0ef47db.json').read_text(encoding='utf8'))
    short={}
    for load in ('decode 1','verify 1'):
        r=[v for v in receipt['benchmarks'] if v['case']['load']==load]
        s=next(v['median_ns']/1000 for v in r if v['case']['layer']=='sliding')
        f=next(v['median_ns']/1000 for v in r if v['case']['layer']=='full')
        short[load]={'sliding_us':s,'full_us':f,'sliding_only_us_required_for_1pct_sixload':(46*s+16*f)*(1-1/1.01**6)/46}
    return {'source_pins':{str(p.relative_to(root)):sha(t) for p,t in zip((external,vp,own,launch),(ex,vs,os,ls))},
            'clean_control_job':'3f44f891d0ef47db','short_sensitivity':short,
            'owned_complete_head_transport':{'owner_rows':4,'senders':16,'row_stride_f32':68,
             'buffer_bytes':16*4*68*4,'meaningful_bytes_per_sender':4*65*4,
             'expected_tx_bytes':16*4*65*4,'with_one_u64_mbarrier_bytes':16*4*68*4+8},
            'v1_logical_partial_payload_not_hbm':{'owned_global_updates_bytes':33*128*65*4,
             'owned_final_read_plus_zero_bytes':2*128*65*4,
             'failed_HC_DSM_payload_bytes':32*64*65*4,'VP2_DSM_payload_bytes':32*16*1056},
            'decision':'No source-only >=1% matched BF16 causal headroom established; requalify exact VP2 BF16 boundary before owned rewrite.',
            'scope':'No GPU; source fingerprints/structural facts/conditional arithmetic only.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    r=inspect(a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
    print(json.dumps(r,indent=2))
