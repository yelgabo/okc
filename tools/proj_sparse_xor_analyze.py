"""Selected C2/SX local code/resource census; no GPU or latency measurement."""
import argparse,ast,hashlib,json,math,re
from pathlib import Path


def entries(path):
    result={}
    for text in re.split(r'(?=\.section\s+\.text\.)',path.read_text()):
        if not text.startswith('.section'):continue
        name=text.split('.text.',1)[1].split(',',1)[0].strip()
        if 'sk_kernelI' not in name:continue
        lines=text.splitlines();labels={};operations=[]
        for line in lines:
            label=re.match(r'\s*(\.L_\w+):',line)
            if label:labels[label[1]]=len(operations)
            match=re.search(r'/\*[0-9a-f]+\*/\s+([^\n]+?;)',line)
            if match:operations.append(match[1].strip())
        normalized=[]
        for op in operations:
            op=op.replace(name,'KERNEL_SELF')
            for label,index in sorted(labels.items(),key=lambda x:-len(x[0])):op=op.replace(label,'AT_INSTRUCTION_'+str(index))
            normalized.append(op)
        result[name]=dict(ops=normalized,hmma=sum('HMMA.' in x for x in operations),
            ffma=sum('FFMA' in x for x in operations),load_u16=sum('LDG' in x and '.U16' in x for x in operations),
            membar=sum('MEMBAR' in x for x in operations),barrier=sum('BAR.' in x for x in operations),
            instructions=len(operations))
    return result


def find(mapping,args):
    ints=[int(x) for x in args[:7]]
    prefix='sk_kernelI'+''.join('Li'+str(x)+'E' for x in ints)+'Lb1ELb0ELi'+args[9]+'ELi1E'
    matches=[(k,v) for k,v in mapping.items() if prefix in k]
    assert len(matches)==1,(prefix,len(matches));return matches[0]


def resources(path,name):
    text=path.read_text();match=re.search("Compiling entry function '"+re.escape(name)+"'.*?(?=ptxas info    : Compiling entry|\\Z)",text,re.S)
    assert match,name
    p=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',match[0])
    regs=re.search(r'Used (\d+) registers, used (\d+) barriers',match[0])
    return dict(regs=int(regs[1]),barriers=int(regs[2]),stack=int(p[1]),spill_stores=int(p[2]),spill_loads=int(p[3]))


def main():
    p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('compiled',type=Path);p.add_argument('original',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    source=a.artifact.read_text(encoding='utf8');ns={}
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_XOR_CONFIG':exec(compile(ast.Module([n],[]),'sx-config','exec'),ns)
        if isinstance(n,ast.FunctionDef) and n.name=='xor_expected_layout':exec(compile(ast.Module([n],[]),'sx-layout','exec'),ns)
    new=entries(a.compiled/'device.sass');old=entries(a.original/'device.sass');rows=[]
    for name,c in ns['_XOR_CONFIG'].items():
        args=c['original_targs'].split(', ');oldname,oldentry=find(old,args)
        row=dict(name=name,variants=[])
        for variant in (0,1):
            va=list(args);va[9]=str(int(args[9])|(4096 if variant else 0));symbol,entry=find(new,va)
            props=resources(a.compiled/'ptxas.log',symbol)
            layout=ns['xor_expected_layout'](name,variant,64)
            row['variants'].append(dict(variant=variant,symbol=symbol,**props,
                **{k:v for k,v in entry.items() if k!='ops'},layout_at_example_pmax64=layout,
                unrounded_register_thread_product=props['regs']*layout['threads'],
                opcode_operand_sha256=hashlib.sha256('\n'.join(entry['ops']).encode()).hexdigest()))
            assert props['stack']==props['spill_stores']==props['spill_loads']==0
            if not variant:
                row['control_matches_prior_C2_compiled']=entry['ops']==oldentry['ops']
                assert row['control_matches_prior_C2_compiled'],name
            else:
                assert entry['hmma']==oldentry['hmma'],name
        rows.append(row)
    audit_path=Path(__file__).resolve().parents[1]/'docs/steering/evidence/2026-11-01-projection-waits/audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf8'));loads={int(k):v for k,v in audit['load_sums_us'].items()}
    target=[r for r in audit['rows'] if 'sk_kernel<' in r['c2_profile']]
    assert len(target)==10
    def factor(fraction):
        return math.prod(loads[m]/(loads[m]-fraction*sum(r['c2_us'] for r in target if r['m']==m)) for m in loads)**(1/6)
    result=dict(artifact_sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),rows=rows,
        c2_job=audit['c2_job'],full30_baseline_score=audit['reconstructed_score'],
        hypothetical_uniform_SK_operator_cut={'three_percent':factor(.03),'five_percent':factor(.05),'ten_percent':factor(.10)},
        scenario_scope='Whole-operator score sensitivity only, NOT measured correction cost or speedup forecast',
        normalization='preserve all opcodes/operands; selected self-symbol renamed and each branch label mapped to its target instruction index',
        scope='CUDA12.8 sm90a static compile only; example pmax64 layout is not measured H200 resource admission or latency')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print([(r['name'],[(v['regs'],v['hmma'],v['instructions']) for v in r['variants']]) for r in rows])


if __name__=='__main__':main()
