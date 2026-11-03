"""No-GPU source/dispatch gates for the isolated decode Lt port."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import dl_proj_lt_port as port

src=Path(sys.argv[1]).read_text(encoding='utf-8')
compact=port.compact_docstrings(src)
def canonical(s):
    tree=ast.parse(s)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
            n=node.body[0]
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
                node.body[0]=ast.Pass()
    return ast.dump(tree,include_attributes=False)
assert canonical(src)==canonical(compact), 'executable drift from docstring compaction'
fn=next(n for n in ast.parse(port.RUNTIME).body if isinstance(n,ast.FunctionDef) and n.name=='_proj')
calls=[]
bf16=object()
ns={'torch':SimpleNamespace(bfloat16=bf16),
    '_dlplt_old_proj':lambda *args:calls.append('old') or 'old'}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual proj wrapper>','exec'),ns)
class X:
    dtype=bf16
    def __init__(self,m,aligned=True,contiguous=True):self.shape=(m,8192);self.aligned=aligned;self.contiguous=contiguous
    def data_ptr(self):return 256 if self.aligned else 258
    def is_contiguous(self):return self.contiguous
state={'dense':{'qkv_gate':SimpleNamespace(shape=(2304,8192)),
                'unpooler':SimpleNamespace(shape=(8192,4096))},
       'dlplt':{key:(lambda x:'lt') for key in port.PINS}}
for layer,m in port.PINS:
    n=state['dense'][layer].shape[0]
    assert ns['_proj'](X(m),layer,n,state)=='lt'
    for x,rows in ((X(m,False),n),(X(m,contiguous=False),n),(X(m),n-1),(X(m+1),n)):
        assert ns['_proj'](x,layer,rows,state)=='old'
state['dlplt']={}
assert ns['_proj'](X(256),'qkv_gate',2304,state)=='old'
print('PASS exact executable/CUDA AST preserved by compaction; actual route wrapper full/slice/shape/alignment/contiguity/baseline gates')
