"""OFF native-slot7 effective-gamma prototype. Source/CPU only; no dispatch.

Derived coefficient format changes must be coupled across native pack/selector/
all consumer literals. Supplied raw weights and fallback formats remain intact.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PIN='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
FLAG='OKC_FORK_GAMMA_EFFECTIVE'
HELPER_PINS={'MOE_CUDA_SRC':'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71',"_MA_ALT_SRCS['moe_helper0']":'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716'}
PRIMITIVE='__device__ __forceinline__ float fork_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }\n'
GAMMA_HELPER=r'''
#ifndef OKC_FORK_GAMMA_EFFECTIVE
#define OKC_FORK_GAMMA_EFFECTIVE 0
#endif
#if OKC_FORK_GAMMA_EFFECTIVE != 0 && OKC_FORK_GAMMA_EFFECTIVE != 1
#error invalid_effective_gamma_format
#endif
__device__ __forceinline__ float fork_gamma(float value) {
#if OKC_FORK_GAMMA_EFFECTIVE
  return value;  // immutable BF16 effective coefficient already rounded in prepare
#else
  return fork_bf16(1.f + value);
#endif
}
'''
HOST=r'''
# OFF source epoch only: not a timing-selected policy or public qualification.
FORK_GAMMA_PRECOMPUTE = @ENABLED@

def _fork_gamma_native_weight(weights):
    raw=weights['post_expert_norm']
    if not FORK_GAMMA_PRECOMPUTE:
        return raw.contiguous()  # exact historical raw-slot contract
    if raw.dtype != torch.bfloat16 or raw.numel()!=4096:
        raise RuntimeError('FORK_GAMMA_FORMAT requires4096 BF16 coefficients')
    if not bool(torch.isfinite(raw).all()):
        raise RuntimeError('FORK_GAMMA_FORMAT finite coefficient proof domain')
    return (raw.float()+1).to(torch.bfloat16).contiguous()
'''


def sha(value):return hashlib.sha256(value.encode()).hexdigest()


def replace_gamma(device):
    prefix='fork_bf16(1.f + ';at=0;changes=[]
    while True:
        a=device.find(prefix,at)
        if a<0:break
        end=a+len('fork_bf16(');depth=1
        while depth:
            depth+=(device[end]=='(')-(device[end]==')');end+=1
        old=device[a:end];read=old[len(prefix):-1]
        assert read.startswith(('__bfloat162float(', '__uint_as_float(')),old
        new='fork_gamma('+read+')'
        changes.append(dict(old=old,new=new))
        device=device[:a]+new+device[end:];at=a+len(new)
    assert len(changes)==16 and device.count(PRIMITIVE)==1
    device=device.replace(PRIMITIVE,PRIMITIVE+GAMMA_HELPER)
    return device,changes


def build(source,enabled=False):
    source=source.replace('\r\n','\n');assert sha(source)==PIN,'unrecognized607 source'
    tree=ast.parse(source);lines=source.splitlines(keepends=True);edits=[];consumer={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in HELPER_PINS:
            key=ast.unparse(node.targets[0]);value=ast.literal_eval(node.value)
            if HELPER_PINS[key]:assert sha(value)==HELPER_PINS[key]
            changed,expressions=replace_gamma(value)
            assert '"""' not in changed
            edits.append((node.lineno-1,node.end_lineno,key+' = r"""'+changed+'"""\n'))
            consumer[key]=dict(before_sha256=sha(value),after_sha256=sha(changed),expressions=expressions)
    assert len(consumer)==2
    prep=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_ma_prepare')
    text=ast.get_source_segment(source,prep)
    raw='weights["post_expert_norm"].contiguous(),  # bf16; kernel forms 1 + w'
    assert text.count(raw)==1
    text=text.replace(raw,'_fork_gamma_native_weight(weights),  # coupled native coefficient format')
    text=text.replace('    dev = weights["w13"].device',
        '    st["ma_gamma_effective"] = FORK_GAMMA_PRECOMPUTE\n    dev = weights["w13"].device',1)
    edits.append((prep.lineno-1,prep.end_lineno,text+'\n'))
    select=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_moe')
    text=ast.get_source_segment(source,select)
    anchor='    ext = state["ma_ext"].get(key)';assert text.count(anchor)==1
    text=text.replace(anchor,
        '    if "ma_gamma_effective" in state:\n'
        '        assert state["ma_gamma_effective"] == FORK_GAMMA_PRECOMPUTE,"FORK_GAMMA_FORMAT state mismatch"\n'
        '    if FORK_GAMMA_PRECOMPUTE:\n'
        '        defines=tuple(defines)+("-DOKC_FORK_GAMMA_EFFECTIVE=1",)\n'+anchor)
    edits.append((select.lineno-1,select.end_lineno,text+'\n'))
    for a,b,value in sorted(edits,reverse=True):lines[a:b]=[value]
    result=''.join(lines)+HOST.replace('@ENABLED@',repr(bool(enabled)))
    header='# okc: problem=muse_decode_layer'
    assert result.count(header)==1
    result=result.replace(header,header+'\n# OFF GAMMA PROTOTYPE: SOURCE/CPU ONLY; NOT GPU OR PUBLIC QUALIFIED.',1)
    ast.parse(result)
    return result,dict(control_sha256=PIN,source_sha256=sha(result),bytes=len(result.encode()),
        enabled=bool(enabled),consumer_literals=consumer,native_slot=7,
        fallback_unchanged=True,raw_input_mutated=False,coefficient_bytes=8192,
        finite_scalar_domain=65280,gpu_execution=False,compile_qualification=False,
        inherited_progress_repairs_pending=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('out',type=Path)
    parser.add_argument('--enable',action='store_true');args=parser.parse_args()
    result,manifest=build(args.source.read_text(encoding='utf-8'),args.enable)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_bytes(result.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({key:manifest[key] for key in ('source_sha256','bytes','enabled')}))
