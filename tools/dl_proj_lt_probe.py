"""Build existing four-arm layer harness with explicit Lt receipt and failclosed gates."""
import ast
from pathlib import Path
import sys
import dl_make_bench_pub as bench

if __name__ == '__main__':
    # Existing harness CLI arguments pass through unchanged; diagnostics only.
    anchor='    for line in sorted(_AB_OUT, key=rank):'
    assert bench.BODY.count(anchor)==1
    bench.BODY=bench.BODY.replace(anchor,'''    for line in state.get('dlplt_report', []):
        _ab_print('ABDECIDE ' + line)
'''+anchor)
    head,tail=bench.BODY.rsplit('    return state',1)
    bench.BODY=head+'''    assert not any(line.startswith(('ABFAIL','ABSTOP','ABCHKFAIL')) for line in _AB_OUT), 'incomplete/failed Lt comparison'
    assert all(_AB_NCHECK.get(n,0)==8 and _AB_NFAIL.get(n,0)==0 for n in _AB_ARMS), 'missing Lt cases'
    return state'''+tail
    # Reference's busiest group is argmax, including ties. Second group is an explicit
    # diagnostic heldout distribution after excluding that first group.
    old='                rank = pairs.argsort(descending=True)\n                base["local_start"].fill_(LOCAL_EXPERTS * int(rank[0 if routing == "busiest" else 1]))'
    new='''                first = int(pairs.argmax())
                alternative = pairs.clone(); alternative[first] = -1
                group = first if routing == 'busiest' else int(alternative.argmax())
                base["local_start"].fill_(LOCAL_EXPERTS * group)'''
    assert bench.BODY.count(old)==1
    bench.BODY=bench.BODY.replace(old,new)
    bench.main()
    ast.parse(Path(sys.argv[1]).read_text(encoding='utf-8'))
