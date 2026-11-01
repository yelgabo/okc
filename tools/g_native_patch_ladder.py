"""Source/CPU-only repair ladder; partial snapshots are NEVER runnable candidates.

Instrument the existing pinned string transformer, retaining every exact edit,
instead of inventing a second repair implementation. No submission is imported.
"""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

UPSTREAM='82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19'
INCUMBENT='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
KEYS=('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']")


def sha(s):return hashlib.sha256(s.encode()).hexdigest()


def load(path):
    spec=importlib.util.spec_from_file_location('_audit_'+path.stem,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def literal(source,key):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==key)
    return ast.literal_eval(node.value),node.lineno


GROUPS=[
 ('scope_and_mfin',"    guard='''",'mixed', ['tiny','tiny2','mid','b4','main'],
  'Compile-time exclusion of unreviewed nonWS/nonWGS fallback; MFIN0 excludes unsafe readiness fallthrough. A different proved finalizer is possible, not flag rollback.'),
 ('bf16_primitive',"    s=once(s,'__device__ __forceinline__ float probe_q",'implementation',['tiny','tiny2','mid','b4','main'],
  'Helper definition only; numerical changes appear in subsequent groups.'),
 ('effective_gamma', '    # Gamma equals', 'required_baseline_parity',['tiny','tiny2','mid','b4','main'],
  'BF16(1+rawBF16) effective parameter. Immutable preparation hoisting may preserve it; no measured cost.'),
 ('gate_up_boundary',"    pattern=r'((?:const )?float gt",'required_stage_boundary',['tiny','tiny2','mid','b4','main'],
  'Round scaled gate/up before nonlinear SiLU. Removing rounds is not a conforming ablation.'),
 ('down_boundary',"    pattern=r'(float v0f?",'required_stage_boundary',['tiny','tiny2','mid','b4','main'],
  'Round scaled GEMM2 before gamma. Coupled to WGS scale/gamma separation below.'),
 ('wgs_scale_boundary','    # WGS gamma','required_stage_boundary',['b4','main'],
  'Separate global scale from gamma so the BF16 round is at the correct edge. Unexecuted WGS branches also exist in helper source.'),
 ('ybf_policy',"    if main:\n        s=once(s,'#define OKC_YBF",'conservative_precision_choice',['b4','main'],
  'Disable extra postgamma BF16 quantization; not the required pregamma round. No certified rollback or isolated cost.'),
 ('skinny_reader_proxy','    # Skinny slot reads','required_protocol_property',['tiny','tiny2','mid'],
  'Retain generic-reader to async-refill proxy lifetime bridge. Actual WS-mode applicability needs selected preprocessor path, not source-count cost.'),
 ('wgs_reader_release','    if main:\n        # WGS ring','required_protocol_property',['b4','main'],
  'All-reader proxy arrivals and NT expected counts are one coupled protocol; cannot test either half alone.'),
 ('wgs_writer_proxy','        # generic xA/xB stores','required_protocol_property',['b4','main'],
  'Generic gather/epilogue stores must publish to TMA including finisher stores made after flush entry.'),
 ('final_counter_strengthening','    # Counter publication','conservative_implementation_choice',['tiny','tiny2','mid','b4','main'],
  'Acq_rel RMW chosen atop retained fences/joins/acquire. Only Tiny/Tiny2 have a separately proved relaxed returning-RMW alternative so far.'),
 ('cooperative_admission','    # No programmatic secondary launch','required_progress_property_with_policy_choice',['tiny','tiny2','mid','b4','main'],
  'Enforce resident-grid progress by checked cooperative launch. PDLoff ordinary same-stream completion is the selected safe policy, not a universal PDL ban.'),
]


def delta(before,after):
    a,b=before.splitlines(),after.splitlines();records=[]
    for op,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
        if op!='equal':
            records.append(dict(op=op,before_lines=[i+1,j],after_lines=[k+1,l],
                before=a[i:j],after=b[k:l],context_before=a[max(0,i-2):i],context_after=a[j:j+2]))
    return records


def flags(source):
    values={};nodes=ast.parse(source).body
    names={'MOE_DEFINES','MOE_TINY_DEFINES','MOE_TINY2_DEFINES','MOE_MID_DEFINES',
           'MOE_TINY_MAX_M','MOE_TINY2_MAX_M','MOE_MID_MAX_M','PDL','MOE_HAS_PDL','MFAB_LAYER','SC6_WGS','DQ7'}
    for n in nodes:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names:
            name=n.targets[0].id
            try:values[name]=ast.literal_eval(n.value)
            except (ValueError,TypeError):
                if name=='MOE_MID_DEFINES':
                    assert ast.unparse(n.value)=="tuple(('@moe_helper0df' if d == '@moe_helper0' else d for d in MOE_MID_DEFINES))"
                    values[name]=tuple('@moe_helper0df' if d=='@moe_helper0' else d for d in values[name])
                else:raise
        elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id in names:
            assert isinstance(n.op,ast.Add);values[n.target.id]+=ast.literal_eval(n.value)
    return values


def gamma_sites(device):
    result=[];at=0
    while True:
        start=device.find('fork_bf16(1.f + ',at)
        if start<0:return result
        end=start+len('fork_bf16(');depth=1
        while depth:
            depth+=(device[end]=='(')-(device[end]==')');end+=1
        function=device.rfind('__device__',0,start)
        header=device[function:device.index('{',function)].replace('\n',' ')
        result.append(dict(line=device[:start].count('\n')+1,expression=device[start:end],
            containing_device_header=header,
            in_selected_short_WS_function=any(name+'(' in header for name in ('ws_helper','skinny_ring_ws'))))
        at=end


def build(root):
    root=Path(root)
    paths={'upstream':root/'.build/today/kel/kernels/muse_decode_layer.py',
           'incumbent':root/'muse_decode_layer/cand/g1b4_stage_release.py',
           'repair':root/'tools/dl_fork_moe.py','stage':root/'tools/dl_g1_stage_retire.py',
           'empty':root/'tools/dl_fork_moe_empty.py'}
    text={k:p.read_text(encoding='utf-8') for k,p in paths.items()}
    assert sha(text['upstream'])==UPSTREAM and sha(text['incumbent'])==INCUMBENT
    repair=load(paths['repair']);stage=load(paths['stage']);empty=load(paths['empty'])
    starts=[(text['repair'][:text['repair'].index(marker)].count('\n')+1,key) for key,marker,*_ in GROUPS]
    assert starts==sorted(starts)
    function=next(n for n in ast.parse(text['repair']).body if isinstance(n,ast.FunctionDef) and n.name=='patch_device')
    class Instrument(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='s' for target in node.targets for t in ast.walk(target)):
                return [node,ast.Expr(value=ast.Call(func=ast.Name(id='_record',ctx=ast.Load()),args=[ast.Constant(node.lineno),ast.Name(id='s',ctx=ast.Load())],keywords=[]))]
            return node
    transformed=Instrument().visit(function);ast.fix_missing_locations(transformed)
    results={}
    for key in KEYS:
        raw,rawline=literal(text['upstream'],key);final,finalline=literal(text['incumbent'],key)
        records=[];previous=raw
        def record(line,value):
            nonlocal previous
            if value==previous:return
            group=max((item for item in starts if item[0]<=line),key=lambda p:p[0])[1]
            records.append(dict(group=group,tool_line=line,before_sha256=sha(previous),after_sha256=sha(value),
                changed_hunks=delta(previous,value)))
            previous=value
        env=dict(vars(repair),_record=record)
        exec(compile(ast.Module(body=[transformed],type_ignores=[]),str(paths['repair']),'exec'),env)
        patched=env['patch_device'](raw,key=='MOE_CUDA_SRC')
        assert patched==repair.patch_device(raw,key=='MOE_CUDA_SRC')
        if key=='MOE_CUDA_SRC':
            retired=stage.patch_device(patched)
            records.append(dict(group='stage_retirement',tool='dl_g1_stage_retire.py::patch_device',
                before_sha256=sha(patched),after_sha256=sha(retired),changed_hunks=delta(patched,retired)))
            assert empty.GUARD not in retired and retired.count(empty.HEAD)==1
            guarded=retired.replace(empty.HEAD,empty.HEAD+empty.GUARD)
            records.append(dict(group='empty_routing',tool='dl_fork_moe_empty.py::HEAD/GUARD',
                before_sha256=sha(retired),after_sha256=sha(guarded),changed_hunks=delta(retired,guarded)))
            assert guarded==final,'unclassified main residual'
        else:assert patched==final,'unclassified helper residual'
        results[key]=dict(upstream_literal_line=rawline,incumbent_literal_line=finalline,
            upstream_literal_sha256=sha(raw),incumbent_literal_sha256=sha(final),
            upstream_literal_bytes=len(raw.encode()),incumbent_literal_bytes=len(final.encode()),
            exact_reconstruction=True,steps=records)
    df=next(n for n in ast.parse(text['upstream']).body if isinstance(n,ast.FunctionDef) and n.name=='_df_patch')
    ns={};exec(compile(ast.Module(body=[df],type_ignores=[]),'<pinned-pure-string-DF-transform>','exec'),ns)
    helper_raw,_=literal(text['upstream'],KEYS[1]);helper_final,_=literal(text['incumbent'],KEYS[1])
    dfraw=ns['_df_patch'](helper_raw)
    prep=next(n for n in ast.parse(text['incumbent']).body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_ma_prepare')
    bank=next(n for n in ast.walk(prep) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="st['ma_w']")
    assert isinstance(bank.value,ast.List)
    return dict(schema=1,audit_reference='0f51f77',current_human_priority='ccb5e19',
        source_only=True,gpu_execution=False,partial_snapshots_are_conforming=False,
        input_files={k:dict(path=str(p.relative_to(root)),lf_sha256=sha(text[k])) for k,p in paths.items()},
        taxonomy=[dict(id=key,classification=kind,active_families=active,requirement=why) for key,_,kind,active,why in GROUPS],
        native_ladders=results,configuration={'upstream':flags(text['upstream']),'incumbent':flags(text['incumbent'])},
        mid_df=dict(upstream_transform_line=df.lineno,transform_sha256=sha(ast.get_source_segment(text['upstream'],df)),
            upstream_actual_source_sha256=sha(dfraw),incumbent_actual_source_sha256=sha(helper_final),
            classification='unproved_protocol_excluded',difference_hunks=delta(dfraw,helper_raw)),
        additional_groups=[
          dict(id='stage_retirement',classification='safe_implementation_recovery',active_families=['b4','main'],evidence='63f5a2a8ed2341d1; already in607e'),
          dict(id='empty_routing',classification='required_bounds_correction',active_families=['b4','main'],evidence='zero-local wgs_stage_at nj0 guard'),
          dict(id='state_scale_ownership',classification='required_layout_ownership',active_families=['all'],evidence='state-local ma_sc6 and defines/pack match; no timed cost measurement'),
          dict(id='df_alias',classification='unproved_protocol_excluded',active_families=['mid'],evidence='different data-sentinel protocol; no isolated cost'),
          dict(id='sc6',classification='retained',active_families=['b4','main'],evidence='both SC6_WGS true; eligible state only')],
        gamma=dict(native_prepare_line=prep.lineno,bank_assignment_line=bank.lineno,
            native_slot7_expression=ast.unparse(bank.value.elts[7]),already_effective=False,
            source_sites={k:gamma_sites(literal(text['incumbent'],k)[0]) for k in KEYS},
            host_coefficient_bytes=8192,raw_weights_must_remain_unchanged=True,
            scope_and_mfin_subscopes={'scope_guard':['all_compiled_native'],'MFIN0':['b4','main']}),
        f042=dict(classification='proved_alternative_to_conservative_strengthening',group='final_counter_strengthening',
            affected=['tiny','tiny2'],required_repairs_removed=False,
            is_original_G0_helper=False,measured_cost=None,
            hold='Inherited ATT7 ordinary-launch progress gap; future common repair to both arms required'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();result=build(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'exact_native_reconstruction':True,'groups':len(result['taxonomy']),
        'steps':{k:len(v['steps']) for k,v in result['native_ladders'].items()}}))
