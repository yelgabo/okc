"""OFF-by-default exact ec565 standalone progress epoch. No production writes or dispatch."""
import argparse,ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
PIN='b35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def function(s,anchor):
 assert s.count(anchor)==1,anchor
 a=s.index(anchor);z=s.index('{',a)+1;depth=1
 while depth:depth+=(s[z]=='{')-(s[z]=='}');z+=1
 return s[a:z]
def prior(root):
 saved=sys.path[:];sys.path.insert(0,str(root/'tools'))
 try:
  sp=importlib.util.spec_from_file_location('_standalone_coop_pattern',root/'tools/moe_owned_cooperative_pair.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m.MAIN,m.SKINNY
 finally:sys.path[:]=saved

def core(source,root):
 assert len(source.encode())==235700 and sha(source)==PIN,'unreviewed historical source'
 main,skinny=prior(root)
 assert sha(main)=='86108ac9962c77c158adb6927cbcefbecb8badb98fdfbae6ba909f467e8c31bf','cooperative main pattern drift'
 assert sha(skinny)=='648bc217697e567c47874ed6ca602a86bbd81c6609d2a6529e83ac092d6c7592','cooperative skinny pattern drift'
 extra='''    static_assert(kBqMax == 64, "STANDALONE_COOP_RESOURCE: queue proof changed");
    TORCH_CHECK(static_cast<int64_t>(sms) * (kBqMax - 1) >=
                    16 * ((static_cast<int64_t>(p.M) * kTopK) / 64 + p.nloc),
                "STANDALONE_COOP_RESOURCE: dynamic B queue capacity");
'''
 needle='    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));'
 assert main.count(needle)==1;main=main.replace(needle,extra+needle,1)
 result=source;changes=[]
 for anchor,new in [('template <int MODE>\nvoid launch_main(',main),('template <int SPLIT>\nvoid launch(',skinny)]:
  old=function(result,anchor);assert '<<<' in old and 'cudaLaunchCooperative' not in old
  result=result.replace(old,new,1);changes.append((new,old))
 old=function(result,'__device__ __forceinline__ bool spin_failed(')
 a='    atomicOr(ctr + 8, code);\n    return true;';assert old.count(a)==1
 new=old.replace(a,'    atomicOr(ctr + 8, code);\n    asm volatile("trap;" ::: "memory");\n    return true;',1)
 result=result.replace(old,new,1);changes.append((new,old))
 reverse=result
 for new,old in reversed(changes):reverse=reverse.replace(new,old,1)
 assert reverse==source,'native delta escaped two launchers/common watchdog'
 return result,dict(input_sha256=PIN,core_sha256=sha(result),native_delta_only_two_launchers_and_watchdog=True,main_launch_sha256=sha(main),skinny_launch_sha256=sha(skinny),qualification_scope='EP8 local_start0/local_count32; 1..512 tokens per original chunk; H200-compatible admitted SM90 grid with dynamic B queue capacity; no circular/nonzero weight bank claim',GPU=False)

def build(root,source):
 result,m=core(source,root)
 assert result.count('name="okc_moe_fused"')==1;result=result.replace('name="okc_moe_fused"','name="okc_moe_ec565_coop_trap_v1"',1)
 # Shadow old entry/prepare/selftest with explicit fixed policy and disabled qualification gates.
 from moe_standalone_progress_body import RUNTIME
 result+='\n'+RUNTIME+'\n_self_test=_spg_selected_selftest\n'
 m.update(policy_choice='Freeze exact abe765 six-load policy, no tuner/bench called by new prepare',static_gate='FAIL: main47spill12/112 ->16/128, skinny unchanged',full_fresh_empty_graph_qualification='DESIGN ONLY; blocked by static gate',normal_entry_OFF=True)
 ast.parse(result)
 return result,m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();raw=subprocess.check_output(['git','-C',str(a.root),'show','ec565a7:muse_moe_w4a16/submission.py']).decode();out,m=build(a.root,raw);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(out.encode());m.update(output_sha256=sha(out),bytes=len(out.encode()),OFF_by_default=True,H200_qualified=False);a.output.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
