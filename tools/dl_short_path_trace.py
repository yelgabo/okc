"""Offline-only short G pool/router->route->expert trace builder. NEVER submit/dispatch.

Exact607 source. No native source/fence/precision flag changes. Four private state
owners, actual natural route equality, three heldout seeds, full graph timing.
Default FRT versus existing serial BF16-pool+FP32-router is only a prospective
candidate. No speed/ordering claim until completed H200 traces and math gates.
"""
import argparse,ast,hashlib,importlib.util
from pathlib import Path
PIN='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'

def build(source,root):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN
    tree=ast.parse(source);n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_sl_tile_after_attention')
    fn=ast.get_source_segment(source,n)
    needle='    if use_ma:\n'
    assert fn.count(needle)==1
    observation="    if state.get('_sp_active',False):state['_sp_capture'].update(xm=xm,ids=ids,lids=lids,wts=wts,moe_backend=moe)\n"
    fn=fn.replace(needle,observation+needle)
    lines=source.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=[fn+'\n'];s=''.join(lines)
    # No CUDA/CPP literal or other function changes.
    old_constants=[x.value for x in tree.body if isinstance(x,ast.Assign) and isinstance(x.value,ast.Constant)]
    new_constants=[x.value for x in ast.parse(s).body if isinstance(x,ast.Assign) and isinstance(x.value,ast.Constant)]
    assert [ast.dump(x) for x in old_constants]==[ast.dump(x) for x in new_constants]
    spec=importlib.util.spec_from_file_location('_short_helpers',root/'tools/dl_g1_layer_gate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    text=(root/'tools/dl_make_bench_pub.py').read_text(encoding='utf8');body=next(ast.literal_eval(x.value) for x in ast.parse(text).body if isinstance(x,ast.Assign) and getattr(x.targets[0],'id','')=='BODY')
    funcs={x.name:ast.get_source_segment(body,x) for x in ast.parse(body).body if isinstance(x,ast.FunctionDef)}
    helpers='\n\n'.join(funcs[k] for k in m.NAMES)
    local=Path(__file__).parent
    out=s+'\n'+m.PRELUDE+'\n'+helpers+'\n'+(local/'dl_g1_layer_owners.py').read_text(encoding='utf8')+'\n'+(root/'tools/dl_timeline_codec.py').read_text(encoding='utf8')+'\n'+(local/'dl_short_path_body.py').read_text(encoding='utf8')
    ast.parse(out);assert len(out.encode())<1_000_000
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=build(a.source.read_text(encoding='utf8'),a.root);a.output.write_bytes(out.encode());print('SHORT_TRACE',len(out.encode()),hashlib.sha256(out.encode()).hexdigest())
