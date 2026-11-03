import ast,io,json,sys,tempfile,tokenize
from pathlib import Path
sys.path.insert(0,'tools')
import g_expert_g224_chunk_phased as generator
from g224_reference_pack import consolidate
from g_expert_g224_prefix import sha
with tempfile.TemporaryDirectory() as directory:
 text,native,_=generator.generate(Path.cwd(),Path(directory))
text=consolidate(text)
license_start=text.index("#                                  Apache License")
license_end=text.index("def _release_private_scratch",license_start)
start_line=text[:license_start].count("\n")+1
end_line=text[:license_end].count("\n")+1
level=0
items=[]
removed_rows=set()
for token in tokenize.generate_tokens(io.StringIO(text).readline):
 if token.type==tokenize.COMMENT and not (token.start[0]<=17 or start_line<=token.start[0]<end_line):
  removed_rows.add(token.start[0])
  continue
 if token.type==tokenize.NL and (not token.line.strip() or token.start[0] in removed_rows):
  continue
 if token.type==tokenize.INDENT:
  level+=1
  items.append((token.type,' '*level))
 elif token.type==tokenize.DEDENT:
  level-=1
  items.append((token.type,token.string))
 else:
  items.append((token.type,token.string))
packed=tokenize.untokenize(items)
assert ast.dump(ast.parse(packed),include_attributes=False)==ast.dump(ast.parse(text),include_attributes=False)
comments=lambda source:[token.string for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type==tokenize.COMMENT]
strings=lambda source:[token.string for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type==tokenize.STRING]
expected_comments=[token.string for token in tokenize.generate_tokens(io.StringIO(text).readline) if token.type==tokenize.COMMENT and (token.start[0]<=17 or start_line<=token.start[0]<end_line)]
assert comments(packed)==expected_comments
assert all(term in packed for term in ("Portions Copyright vjeux", "Submitted by alygg77", "Copyright 2026 team keell0renz", "Permission is hereby granted", "THE SOFTWARE IS PROVIDED"))
assert strings(packed)==strings(text)
print(json.dumps(dict(bytes=len(packed.encode()),sha256=sha(packed),comments=len(comments(packed)),notices_retained=True,exact_string_tokens=True,ast_equal=True,native_sha=sha(native))))
