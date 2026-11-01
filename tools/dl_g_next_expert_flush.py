"""Single-factor Family G screen: defer previous-unit flush until first stage retires.
No ordering primitive or publication operation is removed. No dispatch.
"""
import ast
import hashlib
from pathlib import Path

DEVICE_SHA="c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71"
SEG_SHA="c67020c073a65b7042104903c1f679fbd7230c23d8bcd0e76e69306631340be3"
SIG="template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg("
OLD="        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);"
NEW="  if (pd.kind) wgs_flush<PB>(p, S, pd);  // Previous-unit publication after current first-stage retirement."

def patch_device(cuda):
 if hashlib.sha256(cuda.encode()).hexdigest()!=DEVICE_SHA:raise ValueError("unreviewed WGS device")
 a=cuda.index(SIG);b=cuda.index("\n}\n",a)+2;seg=cuda[a:b]
 assert hashlib.sha256(seg.encode()).hexdigest()==SEG_SHA
 assert seg.count(OLD)==1 and seg.count("  stage(std::true_type{});")==1
 old_comment="""        // the previous segment's deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
"""
 assert seg.count(old_comment)==1
 revised=seg.replace(old_comment+OLD+"\n","")
 revised=revised.replace("  stage(std::true_type{});","  stage(std::true_type{});\n"+NEW)
 return cuda[:a]+revised+cuda[b:]

def patch_source(source):
 t=ast.parse(source);nodes=[n for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="MOE_CUDA_SRC"]
 assert len(nodes)==1;n=nodes[0];cuda=ast.literal_eval(n.value);new=patch_device(cuda)
 lines=source.splitlines(keepends=True);old="".join(lines[n.lineno-1:n.end_lineno]);assert old.count(cuda)==1
 lines[n.lineno-1:n.end_lineno]=[old.replace(cuda,new)];output="".join(lines)
 after=ast.parse(output);next(n for n in after.body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="MOE_CUDA_SRC").value=ast.Constant(cuda)
 assert ast.dump(t)==ast.dump(after)
 return output

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("output");args=p.parse_args()
 s=patch_source(Path(args.input).read_text(encoding="utf8"));Path(args.output).write_text(s,encoding="utf8",newline="\n")
 print(len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
