from pathlib import Path
import re,json,hashlib,collections
r=Path('.build/today-attn/comment-compile');out=[]
def extract(s):
 result={}
 for m in re.finditer(r'(?m)^\s*\.section\s+\.text\.([^,\n]+)',s):
  end=s.find('\n\t.section\t',m.end());v=s[m.end():end if end>=0 else len(s)]
  # Keep actual instructions including predicates, operands, branch labels; discard
  # only address comments/line directives and normalize compiler-generated namespace.
  ins=[]
  for x in v.splitlines():
   q=re.match(r'\s*/\*[0-9a-f]+\*/\s*(.*?)\s*$',x)
   if q and not q[1].startswith('.'):ins.append(re.sub(r'_ZN53_GLOBAL__N__00000000_16_owned_decoder_cu_[0-9a-f]+_\d+', '_GLOBAL_',q[1]))
  key=re.sub(r'_ZN53_GLOBAL__N__00000000_16_owned_decoder_cu_[0-9a-f]+_\d+', '_GLOBAL_',m[1]);result[key]='\n'.join(ins)
 return result
for k in ['one','moe']:
 a=extract((r/(k+'-original')/'device.sass').read_text());b=extract((r/(k+'-comment')/'device.sass').read_text());assert a.keys()==b.keys()
 for name,v in a.items():
  assert v and v==b[name],(k,name)
  counter=collections.Counter(x.split(';')[0].strip().split()[0] for x in v.splitlines())
  row=dict(kind=k,kernel=name,instructions=len(v.splitlines()),normalized_sass_sha256=hashlib.sha256(v.encode()).hexdigest(),exact_instructions_operands_equal=True,special={x:sum(x in line for line in v.splitlines()) for x in ['HGMMA','QGMMA','BAR','MEMBAR','FENCE','LDL','STL']});out.append(row)
 print(k,'equal',[(x['instructions'],x['special']) for x in out if x['kind']==k])
(r/'comparison.json').write_text(json.dumps(out,indent=2))

