"""Family G: two-stage fully drained X-slot retirement; source/compile screen only."""
import ast
import hashlib
from pathlib import Path
from dl_g_next_expert_flush import DEVICE_SHA,SEG_SHA,SIG
OLD="""    // All reads of this activation slot are complete before the ring can reuse it.
    wg_wait<0>();
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    mbar_arrive1(S.bars + BX_E + xs);"""
NEW="""    // Pair retirement: both slots remain owned until all issued reads complete.
    // A final odd stage is drained alone. Per-reader fences remain per released slot.
    if (((seq - first_seq) & 1) || seq - first_seq == s1 - s0 - 1) {
      wg_wait<0>();
      if ((seq - first_seq) & 1) {
        const int previous_xs = (xs + WGS_NX - 1) % WGS_NX;
        asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
        mbar_arrive1(S.bars + BX_E + previous_xs);
      }
      asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
      mbar_arrive1(S.bars + BX_E + xs);
    }"""

def patch_device(cuda):
 if hashlib.sha256(cuda.encode()).hexdigest()!=DEVICE_SHA:raise ValueError("unreviewed stage control")
 a=cuda.index(SIG);b=cuda.index("\n}\n",a)+2;seg=cuda[a:b]
 assert hashlib.sha256(seg.encode()).hexdigest()==SEG_SHA
 assert seg.count(OLD)==1
 assert cuda.count("#define OKC_WGSNX 3")==1 and cuda.count("#define OKC_WGSGS 2")==1
 seg=seg.replace("  auto stage = [&](auto first) {","  const int first_seq = seq;\n  auto stage = [&](auto first) {")
 seg=seg.replace(OLD,NEW)
 return cuda[:a]+seg+cuda[b:]

def patch_source(source):
 tree=ast.parse(source);n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="MOE_CUDA_SRC")
 cuda=ast.literal_eval(n.value);revised=patch_device(cuda)
 lines=source.splitlines(keepends=True);piece="".join(lines[n.lineno-1:n.end_lineno]);assert piece.count(cuda)==1
 lines[n.lineno-1:n.end_lineno]=[piece.replace(cuda,revised)];output="".join(lines)
 after=ast.parse(output);next(n for n in after.body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="MOE_CUDA_SRC").value=ast.Constant(cuda)
 assert ast.dump(tree)==ast.dump(after)
 return output

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("output");a=p.parse_args()
 output=patch_source(Path(a.input).read_text(encoding="utf8"));Path(a.output).write_text(output,encoding="utf8",newline="\n")
 print(len(output.encode()),hashlib.sha256(output.encode()).hexdigest())
