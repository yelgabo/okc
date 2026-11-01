"""OFF source-only common pre-expert coefficient parity repair; exact607 only."""
import argparse, ast, hashlib, json
from pathlib import Path

PIN='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
FIELD='_g_preexp_effective'
CONSUMERS=('_route_b','_route','_route_split','_graft_route','_sl_graft_route','_sl_tile_route','_gle_route')
PREP='    st["g_preexp"] = weights["pre_expert_norm"].contiguous()\n'
DERIVED='    st["_g_preexp_effective"] = (st["g_preexp"].float() + 1.0).to(torch.bfloat16).contiguous()\n'
EDITS=(('1.0 + tl.load(GPRE + c).to(tl.float32)','tl.load(GPRE + c).to(tl.float32)',5),
       ('1.0 + tl.load(GPRE + oc).to(tl.float32)','tl.load(GPRE + oc).to(tl.float32)',1),
       ('1.0+tl.load(GPRE+oc).to(tl.float32)','tl.load(GPRE+oc).to(tl.float32)',2))


def sha(s):return hashlib.sha256(s.encode()).hexdigest()


def census(source):
    tree=ast.parse(source);functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    found={};calls=[]
    for name in CONSUMERS:
        node=functions[name]
        loads=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=='load' and any(isinstance(x,ast.Name) and x.id=='GPRE' for x in ast.walk(n))]
        assert len(loads)==(2 if name=='_sl_tile_route' else 1),(name,len(loads))
        found[name]=dict(line=node.lineno,loads=[n.lineno for n in loads])
    reads=[n for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name)
           and n.value.id=='state' and isinstance(n.slice,ast.Constant) and n.slice.value=='g_preexp']
    assert len(reads)==15 and source.count('state["g_preexp"]')==len(reads),'state consumer census changed'
    for node in reads:
        users=[c for c in ast.walk(tree) if isinstance(c,ast.Call) and node in c.args]
        assert len(users)==1 and isinstance(users[0].func,ast.Subscript),('non-direct consumer',node.lineno)
        kernel=users[0].func.value
        assert isinstance(kernel,ast.Name) and kernel.id in CONSUMERS,(node.lineno,ast.unparse(kernel))
        calls.append(dict(line=node.lineno,kernel=kernel.id))
    assert source.count(PREP)==1 and source.count('GPRE')==sum(ast.get_source_segment(source,functions[n]).count('GPRE')for n in CONSUMERS),'unreviewed GPRE occurrence'
    return dict(consumers=found,callers=sorted(calls,key=lambda x:x['line']),load_count=8,caller_count=15)


def build(source,enabled=False):
    assert sha(source)==PIN,'unreviewed source'
    info=census(source)
    result=source
    if enabled:
        assert FIELD not in source
        for old,new,count in EDITS:
            assert result.count(old)==count,(old,result.count(old))
            result=result.replace(old,new)
        result=result.replace('state["g_preexp"]',f'state["{FIELD}"]')
        result=result.replace(PREP,PREP+DERIVED,1)
        # Exact source reversal is stronger than an outside-AST equality claim.
        reverse=result.replace(PREP+DERIVED,PREP,1).replace(f'state["{FIELD}"]','state["g_preexp"]')
        for old,new,count in EDITS:
            assert reverse.count(new)==count,(new,reverse.count(new))
            reverse=reverse.replace(new,old)
        assert reverse==source,'repair escaped coefficient, callers or prepare assignment'
        ast.parse(result)
    return result,dict(input_sha256=PIN,output_sha256=sha(result),bytes=len(result.encode()),enabled=enabled,
                       field=FIELD,raw_field_retained=True,census=info,default_off=True,
                       source_reverse_proof=bool(enabled),GPU=False,scope='Inherited pre-expert effective coefficient parity only; not integration delta or performance treatment')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--enable',action='store_true');a=p.parse_args()
    s,m=build(a.source.read_text(encoding='utf8'),a.enable);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(s.encode());a.output.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(m))
