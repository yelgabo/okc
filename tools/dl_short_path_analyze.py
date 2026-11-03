"""Analyze complete short-path trace logs/receipts. No isolated-kernel speedup credit."""
import argparse,json,math,re,ast
from pathlib import Path

def analyze(report):
    # Runtime retained every CUDA event; chain summary is redundant metadata.
    source=Path(__file__).with_name('dl_short_path_body.py').read_text(encoding='utf8')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_sp_chain')
    env={};exec(compile(ast.Module(body=[node],type_ignores=[]),'chain','exec'),env)
    for row in report['rows']:
        if 'chain' not in row['trace']:
            events=[(round(a*1000),round(z*1000),stream,name) for a,z,stream,name in row['trace']['events']]
            row['trace']['chain']=env['_sp_chain'](events,row['mode'])
    rows=report['rows'];assert len(rows)==48 and report['checks']==96
    seeds=sorted({r['seed'] for r in rows});assert len(seeds)==3
    results=[]
    for seed in seeds:
        logs=[0.]*4;null=0.;regression=True;deltas=[]
        for length in (1,8):
            cases=[r for r in rows if r['seed']==seed and r['length']==length]
            assert len(cases)==8
            arms=('fused1','fused2','split1','split2')
            med={c:{r['arm']:r['median_us'] for r in cases if r['context']==c} for c in ('sliding','full')}
            cost=[46*med['sliding'][a]+16*med['full'][a] for a in arms]
            null+=max(abs(math.log(cost[0]/cost[1])),abs(math.log(cost[2]/cost[3])))/6
            for k,(c,v) in enumerate(((0,2),(0,3),(1,2),(1,3))):
                logs[k]+=math.log(cost[c]/cost[v])/6
                for ctx in med:regression &= med[ctx][arms[v]]-med[ctx][arms[c]]<=max(.5,.01*med[ctx][arms[c]])
            chains={r['context']+':'+r['arm']:r['trace']['chain'] for r in cases}
            deltas.append({'length':length,'full_graph_medians':med,'graph_chain_intervals':chains})
        passed=min(logs)>=math.log(1.01) and min(logs)>2*null and regression
        results.append({'seed':seed,'sixload_four_cross':[math.exp(x) for x in logs],'duplicate_log_null':null,'regression_gate':regression,'prospective_1percent_gate':passed,'cases':deltas})
    return {'decision_for_further_candidate_gate':all(r['prospective_1percent_gate'] for r in results),'seeds':results,
            'qualifications':'other4loads heldfixed, actual fullgraphs timed; source unchanged outside short producer; pure candidate official/fresh gate still required; no kernel-duration sums credited'}

if __name__=='__main__':
    import importlib.util
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    text=a.log.read_text(encoding='utf8')
    if a.log.suffix=='.json':
        receipt=json.loads(text);assert receipt['status']=='done'
        result=receipt['result'];assert result['passed'] and result['checks']['ok'],'failed/incomplete official receipt'
        tests=[r for r in result['records'] if r.get('event')=='test']
        assert len(tests)==13 and all(r['status']=='pass' for r in tests),'13 official control tests required'
        text=result['log_tail']
    matches=re.findall(r'SHORT_PATH_TRACE ([A-Za-z0-9+/=]+)',text);assert len(matches)==1
    spec=importlib.util.spec_from_file_location('codec',a.root/'tools/dl_timeline_codec.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    print(json.dumps(analyze(m.decode_report(matches[0])),indent=2))
