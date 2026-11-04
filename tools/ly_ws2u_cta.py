"""Per-CTA groups of a --raw MID timeline JSON (tools/ly_box_moe_tl16.py or ly_box_moe_tl16g.py): CTAs whose phase A / B
ranges hold a two-tile job (ranges recomputed from the routing with the build's split weights W1:W2) against the rest.

    python3 tools/ly_ws2u_cta.py RAW.json W1 W2
"""
import json, statistics as st, sys
d=json.load(open(sys.argv[1])); W1,W2=int(sys.argv[2]),int(sys.argv[3])
RB_A,RB_B=512,256
for row in d['rows']:
    for inp in row['inputs'][:3]:
        raw=inp['raw']; lids=inp['lids']
        cnt=[0]*32
        for r in lids:
            for e in r:
                if 0<=e<32: cnt[e]+=1
        jobs=[c for c in cnt if c>0]; two=[c>8 for c in jobs]; G=len(raw)
        def split(rbs):
            w=[W2 if t else W1 for t in two]; C=rbs*sum(w); pre=[0]
            for x in w: pre.append(pre[-1]+rbs*x)
            r=[]
            for c in range(G+1):
                t=C*c//G; u=len(jobs)*rbs
                for j in range(len(jobs)):
                    if pre[j]+rbs*w[j]>t: o=t-pre[j]; u=j*rbs+-(-o//w[j]); break
                r.append(u)
            return r
        rA,rB=split(RB_A),split(RB_B); t0=min(x[0] for x in raw)
        out=[]
        for c,x in enumerate(raw):
            nA=x[17]&0xffff; nB=(x[17]>>16)&0xffff
            a2=any(two[v//RB_A] for v in range(rA[c],rA[c+1])); b2=any(two[v//RB_B] for v in range(rB[c],rB[c+1]))
            ts=[x[18+q] for q in range(46) if x[18+q]]
            ia=[(ts[q]-ts[q-1])/1e3 for q in range(1,min(nA,len(ts)))]
            ib=[(ts[q]-ts[q-1])/1e3 for q in range(nA+1,len(ts))]
            f=lambda k:(x[k]-t0)/1e3
            out.append(dict(c=c,nA=nA,nB=nB,a2=a2,b2=b2,first=f(2),aend=f(3),dep0=f(4),dep1=f(5),bend=f(6),ia=st.median(ia) if ia else 0,ib=st.median(ib) if ib else 0))
        print(row['case']['layer'],"jobs",jobs, "max bend %.2f"%max(o['bend'] for o in out))
        for key in [(False,False),(True,False),(False,True),(True,True)]:
            s=[o for o in out if (o['a2'],o['b2'])==key]
            if not s: continue
            m=lambda k:st.median(o[k] for o in s)
            print("  A2 %d B2 %d n %3d nA %s nB %s first %.2f aend %.2f B-wait %.2f..%.2f bend med %.2f max %.2f intA %.3f intB %.3f"%(key[0],key[1],len(s),sorted(set(o['nA'] for o in s)),sorted(set(o['nB'] for o in s)),m('first'),m('aend'),m('dep0'),m('dep1'),m('bend'),max(o['bend'] for o in s),m('ia'),m('ib')))
