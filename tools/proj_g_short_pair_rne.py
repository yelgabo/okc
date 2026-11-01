"""One OFF G-short representation: paired BF16 RNE at gate/up/down edges only."""
import argparse,ast,hashlib,json
from pathlib import Path
PIN='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
HELPER_PIN='b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716'
FLAG='OKC_SHORT_PAIR_RNE'
PRIMITIVE='__device__ __forceinline__ float fork_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }\n'
PAIR=r'''
#ifndef OKC_SHORT_PAIR_RNE
#define OKC_SHORT_PAIR_RNE 0
#endif
#if OKC_SHORT_PAIR_RNE
#if !OKC_TINY || !OKC_WS || OKC_SPLIT || OKC_WGS || OKC_NW != 8 || OKC_FAST1 != 2
#error SHORT_PAIR_RNE_requires_selected_unsplit_WS_Tiny
#endif
__device__ __forceinline__ float2 fork_pair_rne(float a, float b) {
  return __bfloat1622float2(__floats2bfloat162_rn(a, b));
}
#endif
'''
GATE='const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);'
DOWN='const float v0f = fork_bf16((h ? sm.y : sm.x) * gsc) * nw0, v1f = fork_bf16((h ? sm.w : sm.z) * gsc) * nw1;'
GATE_NEW='const float2 fork_rne = fork_pair_rne((h ? sm.y : sm.x) * gsc, (h ? sm.w : sm.z) * gsc);\n          const float gt = fork_rne.x, up = fork_rne.y;'
DOWN_NEW='const float2 fork_rne = fork_pair_rne((h ? sm.y : sm.x) * gsc, (h ? sm.w : sm.z) * gsc);\n      const float v0f = fork_rne.x * nw0, v1f = fork_rne.y * nw1;'

def sha(s):return hashlib.sha256(s.encode()).hexdigest()

def build(source,enabled=False):
    assert sha(source)==PIN,'unreviewed607 source'
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="_MA_ALT_SRCS['moe_helper0']")
    device=ast.literal_eval(node.value);assert sha(device)==HELPER_PIN
    result=source;edits=[]
    if enabled:
        changed=device
        for old,new,count in ((GATE,GATE_NEW,1),(DOWN,DOWN_NEW,2)):
            assert changed.count(old)==count
            replacement='#if '+FLAG+'\n          '+new+'\n#else\n          '+old+'\n#endif'
            changed=changed.replace(old,replacement);edits.append((old,replacement,count))
        assert changed.count(PRIMITIVE)==1
        changed=changed.replace(PRIMITIVE,PRIMITIVE+PAIR)
        reverse=changed.replace(PRIMITIVE+PAIR,PRIMITIVE)
        for old,new,count in reversed(edits):
            assert reverse.count(new)==count;reverse=reverse.replace(new,old)
        assert reverse==device,'unrelated helper source delta'
        lines=source.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=["_MA_ALT_SRCS['moe_helper0'] = r\"\"\""+changed+'"""\n']
        result=''.join(lines)+('\n# OFF prototype opt-in, fixed Tiny/Tiny2 only; no runtime selection.\nMOE_TINY_DEFINES += ("-D'+FLAG+'=1",)\nMOE_TINY2_DEFINES += ("-D'+FLAG+'=1",)\n')
        ast.parse(result)
    return result,dict(control_sha256=PIN,helper_control_sha256=HELPER_PIN,source_sha256=sha(result),bytes=len(result.encode()),enabled=enabled,
        target='Tiny/Tiny2 T1/T8, unsplit WS, NW8/FAST1=2',source_pairs=dict(gate_up=1,down=2),h_unrolled_pair_sites_per_entry=6,
        gamma_unchanged=True,format_ABI_unchanged=True,all_fences_retained=True,GPU=False,qualification=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--enable',action='store_true');a=p.parse_args();s,m=build(a.source.read_text(encoding='utf8'),a.enable);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(s.encode());a.output.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(m))
