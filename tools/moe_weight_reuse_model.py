"""CPU-only exact skinny job/unit mapping and proposed two-CTA pairing model.

Logical request bytes are NOT HBM bytes: repeated requests can already hit L2.
No CUDA, benchmark dispatch, kernel changes, or projected performance.
Source: submission.py issue_unit and moe_fused_kernel<SPLIT=1>.
"""
from dataclasses import dataclass
from collections import Counter
import argparse
import json

UNIT_BYTES=36864
PHASE_UNITS=(512,256)


@dataclass(frozen=True)
class Unit:
    cta:int
    step:int
    phase:int
    job:int
    expert:int
    rowblock:int

    @property
    def payload(self): return self.phase,self.expert,self.rowblock


def jobs(counts):
    assert len(counts)==32 and all(isinstance(c,int) and c>=0 for c in counts)
    return [(e,k,min(8,c-8*k)) for e,c in enumerate(counts) for k in range((c+7)//8)]


def original_map(counts,ctas=132):
    js=jobs(counts)
    out=[]
    for cta in range(ctas):
        step=0
        for phase,u in enumerate(PHASE_UNITS):
            total=len(js)*u
            for v in range(total*cta//ctas,total*(cta+1)//ctas):
                j,rb=divmod(v,u)
                out.append(Unit(cta,step,phase,j,js[j][0],rb)); step+=1
    return out


def paired_map(counts,clusters=66):
    # Each cluster owns a contiguous range of pair-units. Both ranks traverse the
    # same conceptual steps; only the final nonexistent job is a dummy consumer.
    js=jobs(counts)
    pairs=(len(js)+1)//2
    out=[]
    for cluster in range(clusters):
        step=0
        for phase,u in enumerate(PHASE_UNITS):
            total=pairs*u
            for v in range(total*cluster//clusters,total*(cluster+1)//clusters):
                pair,rb=divmod(v,u)
                for rank in range(2):
                    j=2*pair+rank
                    if j<len(js):
                        out.append(Unit(2*cluster+rank,step,phase,j,js[j][0],rb))
                step+=1
    return out


def summary(counts,clusters=66):
    js=jobs(counts)
    old=original_map(counts,2*clusters)
    new=paired_map(counts,clusters)
    key=lambda x:(x.phase,x.job,x.rowblock)
    assert Counter(map(key,old))==Counter(map(key,new))
    assert all(n==1 for n in Counter(map(key,new)).values())
    credits=Counter(u.job for u in new if u.phase==0)
    assert all(credits[j]==512 for j in range(len(js)))
    def matching_pairs(units):
        slots={(u.cta,u.step):u.payload for u in units}
        return sum(payload==slots.get((cta+1,step)) for (cta,step),payload in slots.items() if cta%2==0)
    requests=len(old)
    unique=len(set(u.payload for u in old))
    old_match=matching_pairs(old)
    new_match=matching_pairs(new)
    return {"counts":counts,"jobs":len(js),"active_experts":sum(c>0 for c in counts),
        "logical_request_bytes":requests*UNIT_BYTES,"unique_payload_bytes":unique*UNIT_BYTES,
        "logical_duplicate_fraction":(requests-unique)/requests if requests else 0,
        "original_neighbor_same_step_pairs":old_match,
        "paired_same_step_multicast_pairs":new_match,
        "paired_saved_request_fraction":new_match/requests if requests else 0,
        "paired_source_request_bytes":(requests-new_match)*UNIT_BYTES,
        "ctas":2*clusters,"phaseA_credits_per_job":512,
        "interpretation":"logical requests only; HBM/L2 traffic and performance unmeasured"}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts',required=True,help='32 comma-separated local expert counts')
    parser.add_argument('--clusters',type=int,default=66)
    args=parser.parse_args()
    print(json.dumps(summary(list(map(int,args.counts.split(','))),args.clusters),indent=2))
