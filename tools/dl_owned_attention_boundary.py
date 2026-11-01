"""Pinned, minimal pre-output-RMS BF16 repair of the historical owned layer.

Known-good cb71 bytes are never edited. No CUDA execution or public action.
"""
import ast
import hashlib
from pathlib import Path
import sys

BASE_SHA='851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a'
PINS={
    '_ATTN_SRC':'53b766244a59c8b7c0c541982f0aec0e1106fc989968874476b9355e20357c34',
    '_WS_SRC':'d04a10d9a722c7c9ec67e80b64cd225a6559ec53cee460c6cd752f25169d1270',
    '_ONE_SRC':'16c6b9e42cc8575ae79a4c2a4292e44efac416ae78ded974f082a82650d2145d',
}
OLD_LOOP='acc[e] *= inv;'
NEW_LOOP='acc[e] = __bfloat162float(__float2bfloat16_rn(acc[e] * inv));'
OLD_PAIR='    acc.x *= inv;\n    acc.y *= inv;\n    const float r = rsqrtf(warp_sum(acc.x * acc.x + acc.y * acc.y) / D + EPS);'
NEW_PAIR='    acc.x = __bfloat162float(__float2bfloat16_rn(acc.x * inv));\n    acc.y = __bfloat162float(__float2bfloat16_rn(acc.y * inv));\n    const float r = rsqrtf(warp_sum(acc.x * acc.x + acc.y * acc.y) / D + EPS);'


def patch_owned_attention(source):
    assert hashlib.sha256(source.encode()).hexdigest()==BASE_SHA,'Unreviewed owned layer source'
    tree=ast.parse(source)
    edits=[];found=set()
    for node in tree.body:
        if not isinstance(node,ast.Assign):continue
        name=getattr(node.targets[0],'id','')
        if name not in PINS:continue
        assert name not in found
        found.add(name)
        old=ast.literal_eval(node.value)
        assert hashlib.sha256(old.encode()).hexdigest()==PINS[name]
        if name=='_ONE_SRC':
            assert old.count(OLD_LOOP)==2
            new=old.replace(OLD_LOOP,NEW_LOOP)
        else:
            assert old.count(OLD_PAIR)==1
            new=old.replace(OLD_PAIR,NEW_PAIR)
        assert "'''" not in new
        edits.append((node.value,new))
    assert found==set(PINS)
    # Only literal value spans are changed. All other source bytes survive.
    lines=source.splitlines(keepends=True)
    starts=[];offset=0
    for line in lines:starts.append(offset);offset+=len(line)
    for value,new in sorted(edits,key=lambda e:e[0].lineno,reverse=True):
        a=starts[value.lineno-1]+value.col_offset
        z=starts[value.end_lineno-1]+value.end_col_offset
        assert lines[value.lineno-1].isascii() and lines[value.end_lineno-1].isascii()
        source=source[:a]+"r'''"+new+"'''"+source[z:]
    after=ast.parse(source)
    for t in (tree,after):
        for n in t.body:
            if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in PINS:
                n.value=ast.Constant('REVIEWED_DEVICE_LITERAL')
    assert ast.dump(tree,include_attributes=False)==ast.dump(after,include_attributes=False)
    return source


if __name__=='__main__':
    src,dst=map(Path,sys.argv[1:])
    result=patch_owned_attention(src.read_text(encoding='utf-8'))
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(result,encoding='utf-8',newline='\n')
    print('OWNED_ATTN_BF16',len(result.encode()),hashlib.sha256(result.encode()).hexdigest())
