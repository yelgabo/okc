"""Actual disassembled CUBIN entry and innermost backedge-loop accounting."""
import collections
import json
from pathlib import Path
import re
import sys


def analyze(sass):
    result={}
    for sec in re.split(r'(?=\n\s*\.section\s+\.text\.)',sass):
        head=sec[:600]
        key='main51' if 'moe_main_kernelILi51' in head else 'main47' if 'moe_main_kernelILi47' in head else 'skinny8w3' if 'moe_fused_kernel' in head else None
        if key is None:continue
        labels={};inst=[]
        for line in sec.splitlines():
            label=re.match(r'\s*(\.L_x_\d+):',line)
            if label:labels[label[1]]=len(inst)
            if re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?[A-Z][A-Z0-9_.]*',line):inst.append(line.strip())
        def counts(lines):
            c=collections.Counter(re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)',l)[1] for l in lines)
            return dict(total=len(lines),opcodes=dict(sorted(c.items())),
                        wait1=sum('WARPGROUP.DEPBAR.LE' in x and ', 0x1 ' in x for x in lines),
                        wait0=sum('WARPGROUP.DEPBAR.LE' in x and ', 0x0 ' in x for x in lines))
        loops=[]
        for i,line in enumerate(inst):
            branch=re.search(r'\bBRA[^;]*\((\.L_x_\d+)\)',line)
            if not branch or branch[1] not in labels:continue
            a=labels[branch[1]]
            if a>=i or i+1-a>1500:continue
            chunk=inst[a:i+1]
            mathlines=[x for x in chunk if 'HGMMA.' in x or 'HMMA.' in x]
            if not mathlines:continue
            loops.append((a,i,mathlines,chunk))
        inner=[]
        for a,b,mathlines,chunk in loops:
            if any(a<=c and d<=b and (a,b)!=(c,d) for c,d,*_ in loops):continue
            item=counts(chunk)
            item.update(start=chunk[0].split('*/')[0][2:],end=chunk[-1].split('*/')[0][2:],
                        math_op=re.search(r'(HGMMA|HMMA)\S*',mathlines[0])[0],math_count=len(mathlines))
            inner.append(item)
        result[key]=dict(entry=counts(inst),hotloops=inner)
    assert set(result)=={'main47','main51','skinny8w3'}
    return result


if __name__=='__main__':
    src,dst=map(Path,sys.argv[1:]);r=analyze(src.read_text())
    dst.write_text(json.dumps(r,indent=2)+'\n')
    for name,item in r.items():
        print(name,item['entry']['total'],'wait1/0',item['entry']['wait1'],item['entry']['wait0'])
        print('loops',[(x['math_op'],x['total'],x['math_count']) for x in item['hotloops']])
