"""SEPARATE owned verify51 launch epoch. Never mutates frozen baselines or dispatches."""
import argparse,ast,hashlib,importlib.util,json,sys
from pathlib import Path
SOURCE_PINS={'384933697aadb2ebfc2abe672b216751e55fd5949e01c0c57d03caa586fe6fe3','c1c6065a095efb407a15c8a1fe8f6936fc3078750b1ca7adb27447846342205e','7612276301d5da296d35995f852b51328f6f33d20c0fd7b9c17bec5fbfe8656c'}
DEVICE_PINS={'35fc183846d5b9bf23efb5c3f0d0f7f556b00bfbe5ea45a3c30c59a9a054b300','aa7b24a3d9a4ba3a6ae4e9480167b569d95bc9f2e921501f4e25b505f07a3c23'}

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def literal(s,name):return next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)
def native_function(s,anchor):
 assert s.count(anchor)==1,anchor
 a=s.index(anchor);z=s.index('{',a)+1;depth=1
 while depth:depth+=(s[z]=='{')-(s[z]=='}');z+=1
 return s[a:z]
def visible(s):
 q=next(q for q in ("'''",'"""') if q not in s);out='r'+q+s+q;assert ast.literal_eval(out)==s;return out
def replace_node(s,node,revised):
 lines=s.splitlines(keepends=True);starts=[0]
 for line in lines:starts.append(starts[-1]+len(line))
 a=starts[node.lineno-1]+len(lines[node.lineno-1].encode()[:node.col_offset].decode())
 z=starts[node.end_lineno-1]+len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
 return s[:a]+revised+s[z:]
def cooperative_launcher(root):
 saved=sys.path[:];sys.path.insert(0,str(root/'tools'))
 try:
  p=root/'tools/dl_fork_owned_cooperative.py';sp=importlib.util.spec_from_file_location('_verify_coop_reuse',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m.LAUNCH
 finally:sys.path[:]=saved

def patch(source,root):
 """Return new source and exact delta manifest; supported original/comment epochs only."""
 assert sha(source) in SOURCE_PINS,'unreviewed owned Python source epoch'
 tree=ast.parse(source);device=literal(source,'_MOE1_SRC');assert sha(device) in DEVICE_PINS,'unreviewed owned device epoch'
 original_device=device;launch=cooperative_launcher(root)
 assert sha(launch)=='38026abe7a6f3c70dc36980317c17d62e82e3823f9a78de16246c9e7502b2d28','unreviewed cooperative helper'
 assert 'cudaLaunchCooperativeKernel' in launch and 'TORCH_CHECK(!p.pdl' in launch
 oldlaunch=native_function(device,'template <typename Kern>\nvoid moe1_launch(')
 assert 'kern<<<grid, block, smem, stream>>>(p);' in oldlaunch and 'cudaLaunchAttributeProgrammaticStreamSerialization' in oldlaunch
 device=device.replace(oldlaunch,launch.rstrip(),1)
 oldwait=native_function(device,'__device__ __forceinline__ bool spin_failed(')
 anchor='    atomicOr(ctr + 8, code);\n    return true;'
 assert oldwait.count(anchor)==1
 newwait=oldwait.replace(anchor,'    atomicOr(ctr + 8, code);\n    asm volatile("trap;" ::: "memory");\n    return true;',1)
 device=device.replace(oldwait,newwait,1)
 assert device.replace(launch.rstrip(),oldlaunch,1).replace(newwait,oldwait,1)==original_device,'device delta escaped scope'
 node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_MOE1_SRC')
 result=replace_node(source,node,visible(device))
 tree=ast.parse(result);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
 calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_moe1_run']
 assert len(calls)==1 and len(calls[0].args)==7 and isinstance(calls[0].args[-1],ast.Constant) and calls[0].args[-1].value is True
 assert ast.literal_eval(calls[0].args[-2])==10,'not selected MAIN51'
 result=replace_node(result,calls[0].args[-1],'False')
 # Distinct native module epoch, so no original extension can satisfy its cache.
 tree=ast.parse(result);load=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_moe1_load')
 name=[k.value for n in ast.walk(load) if isinstance(n,ast.Call) and getattr(n.func,'id','')=='load_inline' for k in n.keywords if k.arg=='name']
 assert len(name)==1 and ast.literal_eval(name[0])=='okc_decode_layer_moe1'
 result=replace_node(result,name[0],repr('okc_decode_layer_moe1_coop_trap_v1'))
 # Actual source emitter must reject every alternative ordinal.
 emitter=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_moe1_source')
 assert 'case 10: launch_main<51>' in ast.get_source_segment(source,emitter)
 ast.parse(result)
 manifest=dict(input_sha256=sha(source),output_sha256=sha(result),input_bytes=len(source.encode()),output_bytes=len(result.encode()),device_before_sha256=sha(original_device),device_after_sha256=sha(device),reused_launcher_sha256=sha(launch),changes=['exact existing checked cooperative host launcher','kernel_fn selected MAIN51 call PDLFalse','spin_failed timeout traps before return','distinct extension module name'],device_delta_only_host_launcher_and_timeout=True,GPU=False,H200_qualified=False)
 return result,manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();s,m=patch(a.source.read_text(encoding='utf8'),a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(s.encode());a.output.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
