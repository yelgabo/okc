"""Generate CHECK-ONLY mode51 producer/memory attribution, no public candidates.

Only full-context verify16/32 are timed. Probes keep stage handshakes/fences but
omit math and have wrong outputs. Probe builds compile only instance8/51 to
fit preparation budget; untouched control/control2 remain full normal builds.
"""
from pathlib import Path
import ast
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent


# Reviewed diagnostic-only delta: pure6da000f -> instrumentedf2b7dfa.
HOOK_PATCHES = (('// (the producer-bound pace).\n', '// (the producer-bound pace).\n// 4/5 (bench-only q_len8 producer builds): retain every acquire/release,\n// proxy fence, warpgroup fence/commit/wait and named barrier, omit tile math;\n// 4 reads real cached rows, 5 zero-fills them. Wrong outputs, never candidates.\n'), ('            constexpr uint32_t CPB = ONE_PROBE == 1 || ONE_PROBE == 3 ? 0u : 16u;\n', '            constexpr uint32_t CPB = ONE_PROBE == 1 || ONE_PROBE == 3 || ONE_PROBE == 5 ? 0u : 16u;\n'), ('            if constexpr (XPIPE != 0) {\n', '            if constexpr (PROD && (ONE_PROBE == 4 || ONE_PROBE == 5)) {\n                int st = cst;\n                for (int it = 0; it < ntiles; ++it) {\n                    float s_probe[32] = {};\n                    const bool common = DCARRY && it < it_c;\n                    acquire(it, st, common);\n                    // Same collective sequence as the current promoted QKOV loop;\n                    // empty wgmma groups retire trivially, no raw-stage readers.\n                    wg_fence();\n                    wg_commit();\n                    wg_wait<0>();\n                    reg_fence(s_probe);\n                    if constexpr (EREL) {\n                        release(st);\n                    }\n                    fence_proxy_async();\n                    named_sync(2 + hk, 128);\n                    wg_fence();\n                    wg_commit();\n                    wg_wait<0>();\n                    reg_fence(o);\n                    if constexpr (!EREL) {\n                        release(st);\n                    }\n                    st = cst;\n                }\n            } else if constexpr (XPIPE != 0) {\n'), ('#if ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)\n', '#if defined(ONE_ATTRIB_ONLY) && ONE_ATTRIB_ONLY\n#define ONE_INSTANCES(X) X(8, 51)\n#elif ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)\n'))

def patch_probe_source(source):
    """Fail closed on source drift or already instrumented input."""
    for old, new in HOOK_PATCHES:
        assert source.count(old) == 1, ('diagnostic hook pattern count', old[:100])
        source = source.replace(old, new, 1)
    return source


def patch_generated_source(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_ONE_SRC' for t in n.targets)]
    assert len(nodes) == 1, 'exactly one inlined owned source required'
    node = nodes[0]
    patched = patch_probe_source(ast.literal_eval(node.value))
    lines = source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['_ONE_SRC = ' + repr(patched) + '\n']
    return ''.join(lines)

GATE=r'''
def _q8_resource_gate():
    mods={"ctrl":_ONE,**_AB_MODS}
    required={"ctrl","ctrl2","read_probe","zero_probe","nocopy_probe","stamp_probe"}
    assert set(mods)==required,f"Missing attribution arm: {required-set(mods)}"
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
    src=patch_generated_source(out.read_text())
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
