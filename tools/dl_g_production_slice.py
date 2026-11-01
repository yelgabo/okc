"""Transparent whole-production G dependency slice; no native-body transform.

Versioned eager bindings preserve wrapper aliases. Function bodies resolve
globals against final module bindings; subscript mutations remain attached to
their container. Original statement text/order, imports and source notices stay.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PIN='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'


def names(node):
    return {n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}


def targets(node):
    roots=node.targets if isinstance(node,ast.Assign) else [node.target]
    result=set()
    for target in roots:
        for n in ast.walk(target):
            if isinstance(n,ast.Name):result.add(n.id)
    return result


def reduce_module(source,roots=('prepare','kernel_fn')):
    tree=ast.parse(source);prior={};final={};dependencies={};deferred={};mutations={}
    for i,node in enumerate(tree.body):
        eager=set();late=set();written=set()
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            written={node.name};late=set().union(*(names(n) for n in node.body))
            eager=set().union(*(names(n) for n in node.decorator_list+[node.args]))
        elif isinstance(node,ast.ClassDef):
            written={node.name};late=set().union(*(names(n) for n in node.body))
            eager=set().union(*(names(n) for n in node.bases+node.decorator_list))
        elif isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            written=targets(node);eager=names(node)
            if isinstance(node,ast.AugAssign):eager|=written
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Subscript) for t in node.targets):
                for name in written:mutations.setdefault(name,set()).add(i)
                eager|=written
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            written={a.asname or (a.name.split('.')[0] if isinstance(node,ast.Import) else a.name) for a in node.names}
        else:eager=names(node)
        dependencies[i]={prior[n] for n in eager if n in prior}
        deferred[i]=late
        for name in written:prior[name]=i;final[name]=i
    # Every version of a mutable container is part of its mutation history.
    for i,node in enumerate(tree.body):
        for name in names(node):
            dependencies[i]|=mutations.get(name,set())-{i}
        dependencies[i]|={final[n] for n in deferred[i] if n in final}
        # Known source uses membership tests such as "_wgs_on" in globals().
        if 'globals' in names(node):
            dependencies[i]|={final[n.value] for n in ast.walk(node)
                              if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value in final}
    kept={final[n] for n in roots}
    kept|={i for i,n in enumerate(tree.body) if isinstance(n,(ast.Import,ast.ImportFrom,ast.Expr))}
    todo=list(kept)
    while todo:
        for dep in dependencies[todo.pop()]-kept:kept.add(dep);todo.append(dep)
    lines=source.splitlines(keepends=True)
    bodies=[];removed=[];cursor=0
    for i,node in enumerate(tree.body):
        start=min([node.lineno]+[n.lineno for n in getattr(node,'decorator_list',[])])
        # Keep every inter-statement comment, including the full mid-file NOTICE
        # and license trailer before the appended production self-test.
        bodies.append(''.join(lines[cursor:start-1]))
        text=''.join(lines[start-1:node.end_lineno])
        if i in kept:bodies.append(text)
        else:removed.append(dict(line=node.lineno,kind=type(node).__name__,
            name=getattr(node,'name',ast.unparse(node.targets[0]) if isinstance(node,ast.Assign) else ''),bytes=len(text.encode())))
        cursor=node.end_lineno
    bodies.append(''.join(lines[cursor:]))
    result=''.join(bodies)
    ast.parse(result)
    return result,removed


def literal_inventory(source):
    result={}
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>1000:
            label=ast.unparse(n.targets[0]);value=n.value.value.encode()
            result[label]=dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
    return result


def fold_disabled_gle(source):
    """Remove only a branch whose sole flag is an unmodified empty dict.

    This does not disable the independent, active GLE_FRT pool/router path.
    """
    tree=ast.parse(source)
    stores=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='GLE_T' and isinstance(n.ctx,ast.Store)]
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='GLE_T' for t in n.targets))
    assert len(stores)==1 and isinstance(assignment.value,ast.Dict) and not assignment.value.keys
    parents={id(child):n for n in ast.walk(tree) for child in ast.iter_child_nodes(n)}
    reads=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='GLE_T' and isinstance(n.ctx,ast.Load)]
    assert len(reads)==2
    for read in reads:
        parent=parents[id(read)]
        assert isinstance(parent,ast.Attribute) and parent.attr=='get'
    fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_after_attention'][-1]
    assert len(fn.body)==2 and isinstance(fn.body[0],ast.If) and isinstance(fn.body[1],ast.Return)
    test=fn.body[0].test
    assert isinstance(test,ast.BoolOp) and isinstance(test.op,ast.And)
    assert ast.unparse(test.values[0])=='GLE_T.get(T, GLE_T.get(str(T), False))'
    assert ast.unparse(fn.body[1].value)=='_gle_prior_after(data, state, o_in, gate, T)'
    lines=source.splitlines(keepends=True);dead=fn.body[0]
    output=''.join(lines[:dead.lineno-1]+lines[dead.end_lineno:])
    return output,dict(flag='GLE_T',value={},reads='two get calls in sole short-circuit condition',
        removed_lines=[dead.lineno,dead.end_lineno],active_GLE_FRT_preserved=True)


def build(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN,'G production source pin'
    folded,fold_proof=fold_disabled_gle(source)
    output,removed=reduce_module(folded)
    before=literal_inventory(source);after=literal_inventory(output)
    assert all(before[k]==v for k,v in after.items()),'native literal drift'
    return output,dict(source_pin=PIN,bytes=len(output.encode()),sha256=hashlib.sha256(output.encode()).hexdigest(),
        removed=removed,literals=after,removed_literals={k:v for k,v in before.items() if k not in after},fold_proof=fold_proof,
        roots=['prepare','kernel_fn'],scope='Whole original prepare and all12 incumbent input routes, no native transforms',
        proof_status='Conservative static dependency prototype; independent source/runtime gates required',H200_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    text,manifest=build(a.source.read_text(encoding='utf8'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(text.encode())
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('literals','removed')},indent=2))
