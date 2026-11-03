"""Compose the separately reviewed owned progress epoch into readable private/production packages."""
import argparse,ast,json
from pathlib import Path
from verify_owned_progress import patch,replace_node,sha

def python_literal(s):
 text='"""\\\n'+s.replace('\\','\\\\').replace('"""','\\"\\"\\"')+'"""';assert ast.literal_eval(text)==s;return text

def compose(source,root):
 tree=ast.parse(source)
 names=('_PAIR_SOURCE_owned','_VERIFY_HYBRID_SOURCE_OWNED')
 targets=[n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in names]
 assert len(targets)==1,'not an exact owned package layout'
 node=targets[0];old=ast.literal_eval(node.value);new,manifest=patch(old,root)
 out=replace_node(source,node.value,python_literal(new))
 if node.targets[0].id=='_PAIR_SOURCE_owned':
  pins=next(n for n in ast.parse(out).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_PAIR_PINS')
  values=ast.literal_eval(pins.value);assert values['owned']==sha(old);values['owned']=sha(new);out=replace_node(out,pins.value,repr(values))
 else:
  mask=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_VERIFY_OVERRIDE_MASK');assert mask in (4,15),'unreviewed production mask'
 ast.parse(out)
 return out,dict(package_input_sha256=sha(source),package_output_sha256=sha(out),package_input_bytes=len(source.encode()),package_output_bytes=len(out.encode()),only_owned_backend_and_private_pin_change=True,backend=manifest,first_two_G_payloads_unchanged=True,GPU=False,H200_qualified=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();out,m=compose(a.source.read_text(encoding='utf8'),a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(out.encode());a.output.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
