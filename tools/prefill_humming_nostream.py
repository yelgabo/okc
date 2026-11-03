"""Single-factor routed INDEXED candidate: disable nondeterministic Stream-K.

Pinned0.1.12 forward.py110 serializes explicit tuning dict; kernel/humming.py
505-513 merges explicit tuning into KernelConfig; config/config.py337 declares
use_stream_k. No heuristic shape, precision, zeroing or routing change.
"""
import ast
from pathlib import Path
import sys
from prefill_humming_indexed import build as indexed_build


def build(src,policy="routed"):
    out=indexed_build(src,policy)
    anchor='        ws={"rank":torch.empty((r,),dtype=torch.int32,device=dev),'
    assert out.count(anchor)==1
    out=out.replace(anchor,'        for config in cfg: config["use_stream_k"]=False\n'+anchor)
    ast.parse(out)
    return out


if __name__=='__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text(),
                                     sys.argv[3] if len(sys.argv)>3 else "routed"))
