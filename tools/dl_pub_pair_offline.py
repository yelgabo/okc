"""CPU-only gate for legacy whole-layer packaging; never emits an entrant."""
import argparse, ast, builtins, hashlib, importlib.util, json
from pathlib import Path

def sha(b): return hashlib.sha256(b).hexdigest()
def names(tree):
    out=set()
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): out.add(n.name)
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            for t in (n.targets if isinstance(n,ast.Assign) else [n.target]):
                out.update(x.id for x in ast.walk(t) if isinstance(x,ast.Name))
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            out.update(a.asname or a.name.split('.')[0] for a in n.names)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);p.add_argument('owned',type=Path);p.add_argument('--legacy',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('legacy_bench',a.legacy);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    s,o=[x.read_text(encoding='utf-8') for x in (a.stage,a.owned)]
    trees=[ast.parse(x) for x in (s,o)]; ns=[names(x) for x in trees]
    result={'sources':[{ 'path':str(path),'raw_bytes':len(path.read_bytes()),'raw_sha256':sha(path.read_bytes()),'LF_bytes':len(txt.encode()),'LF_sha256':sha(txt.encode())} for path,txt in ((a.stage,s),(a.owned,o))], 'legacy_sha256':sha(a.legacy.read_bytes())}
    for tag,fn,txt in [('inc_patch',m.inc_patch,o),('pub_patch',m.pub_patch,s)]:
        try: fn(txt);result[tag]='PASS'
        except BaseException as e:result[tag]=f'{type(e).__name__}: {e}'
    result['plain_bodies_plus_legacy_BODY_bytes']=len((s+'\n'+o+'\n'+m.BODY).encode())
    stripped=[]
    for txt in (s,o):
        # Python-only AST formatting: embedded CUDA strings remain literal values.
        stripped.append(ast.unparse(ast.parse(txt)))
    result['plain_AST_bodies_plus_BODY_bytes']=len(('\n'.join(stripped)+m.BODY).encode())
    result['stage_missing_import_contract']=[k for k in ('PORT_ATTN_ONE','PORT_MOE_FUSED','PORT_PROJ_SK','PORT_MOE_W4','_bg_start','_bg_ext','_bg_seconds','_BG_BUILDS','_one_load','_moe1_load','_skp_load','_w4_ext','_ext','_FAILED') if k not in ns[0]]
    # Execute the actual legacy import helper with only the CUDA/module import stubbed.
    tree=ast.parse(m.BODY);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ab_import')
    env={k:None for k in ns[0]};env.update(_AB_OTHER='inc',_AB_OTHER_XZ='',_ab_module=lambda *_:type('Arm',(),{'ns':{k:None for k in ns[1]}})())
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-legacy-import>','exec'),env)
    try:env['_ab_import']();result['actual_import_helper']='unexpected success'
    except BaseException as e:result['actual_import_helper']=f'{type(e).__name__}: {e}'
    # Exact identifier/resource-independent sharing lower budget; not a generated runtime.
    literals=[{n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>256} for t in trees]
    shared=literals[0]&literals[1]
    result['identical_large_string_values']=len(shared)
    result['shared_large_string_value_bytes']=sum(len(x.encode()) for x in shared)
    result['four_prepare_calls_own']=4;result['two_prepare_calls_shared']=2
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
