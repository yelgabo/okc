"""Source assignment inventory; does not claim an observed H200 state dump."""
import argparse,ast,hashlib,json
from pathlib import Path

def inventory(root):
 p=root/'muse_decode_layer/cand/g1b4_control_release.py';raw=p.read_bytes();source=raw.decode();tree=ast.parse(source);rows=[]
 class Visitor(ast.NodeVisitor):
  def __init__(self):self.scope=[]
  def visit_FunctionDef(self,n):
   self.scope.append(n.name);self.generic_visit(n);self.scope.pop()
  def visit_Assign(self,n):
   for target in n.targets:
    if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id in ('st','state') and isinstance(target.slice,ast.Constant) and isinstance(target.slice.value,str):
     rows.append(dict(scope='.'.join(self.scope),key=target.slice.value,line=n.lineno,expression=ast.unparse(n.value)))
   self.generic_visit(n)
 Visitor().visit(tree)
 return dict(scope='Static reachable-and-conditional assignment sites; not runtime state or completeness proof. Native-eligible path is separately represented in the CPU fixture.',source=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),assignments=rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.write_text(json.dumps(inventory(a.root),indent=2)+'\n',encoding='utf8')
