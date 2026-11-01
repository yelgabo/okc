"""Generate CHECK-ONLY mode51 producer/memory attribution, no public candidates.

Only full-context verify16/32 are timed. Probes keep stage handshakes/fences but
omit math and have wrong outputs. Probe builds compile only instance8/51 to
fit preparation budget; untouched control/control2 remain full normal builds.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent

GATE=r'''
def _q8_resource_gate():
    mods={"ctrl":_ONE,**_AB_MODS}
    for name,mod in mods.items():
        lim=mod.attn_one_prepare()
        rows=[lim[i:i+5] for i in range(0,len(lim),5)]
        found=[r for r in rows if r[:2]==[8,51]]
        assert len(found)==1,(name,rows)
        row=found[0]
        assert row[3:]==[107104,2],f"Q8 resource mismatch {name}: {row}"
        print(f"Q8RESOURCE {name} L,mode,resident,dynamic_bytes,blocks_per_sm={row}",flush=True)
_q8_resource_gate()

'''


def main():
    out=Path(sys.argv[1])
    subprocess.run([sys.executable,str(ROOT/"tools/attn_make_bench_ab.py"),str(out),
        "--null","--shapes","full","--inputs","2","--rounds","120","--seed","311109",
        "read_probe@own=-DONE_PROBE=4,-DONE_ATTRIB_ONLY=1",
        "zero_probe@own=-DONE_PROBE=5,-DONE_ATTRIB_ONLY=1",
        "nocopy_probe@own=-DONE_PROBE=1,-DONE_ATTRIB_ONLY=1",
        "stamp_probe@own=-DONE_STAMP=1,-DONE_ATTRIB_ONLY=1"],check=True,cwd=ROOT)
    src=out.read_text()
    first,rest=src.split("\n",1)
    src=first+"\n# CHECK ONLY: wrong-output attribution probes; never score.\n"+rest
    marker="        yield batch, layer, q_len"
    assert src.count(marker)==1
    src=src.replace(marker,"        if q_len != 8:\n            continue\n"+marker)
    # The exact target shapes, not unsupported modes in probe-only8/51 modules.
    extra="    shapes += [_OB_TESTS[i] for i in (0, 1, 2, 3, 7, 9, 10)]"
    assert src.count(extra)==1
    src=src.replace(extra,"    # Q8 attribution: target verify16/32 full only.")
    marker="# Own-state arms: a private _ONE_STATE dict per arm, swapped in around each call."
    assert src.count(marker)==1
    src=src.replace(marker,GATE+marker)
    # Repeat gate metadata at the tail so it survives receipt truncation.
    marker="            _ab_stsum()"
    assert src.count(marker)==1
    src=src.replace(marker,"            _q8_resource_gate()\n"+marker)
    out.write_text(src)


if __name__=="__main__": main()
