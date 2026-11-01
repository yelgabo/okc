"""Lexer/notice negatives and independent Clang preprocessor tokens; CPU only."""
import ast,re,subprocess,unittest
from pathlib import Path
from prod_size_comments import strip_cpp,python_comments,process

ZIG=Path('/root/.cache/okc-host-tools/zig-0-15-2/ziglang/zig')

def preprocessed_tokens(source):
 p=subprocess.run([str(ZIG),'c++','-E','-x','c++','-std=c++17','-Xclang','-dump-tokens','-'],input=source,capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 result=[]
 for line in p.stderr.splitlines():
  if 'Loc=<' not in line:continue
  token=line.split('Loc=<',1)[0].strip()
  token=re.sub(r'\s*\[(?:StartOfLine|LeadingSpace)\]','',token).strip()
  result.append(token)
 assert result,'no actual preprocessor tokens'
 return result

class Comments(unittest.TestCase):
 def test_quotes_raw_char_escapes_and_token_adjacency(self):
  cases=[r'''const char* a="slash //not"; const char* b="quote\"/*not*/"; // remove
int x=1;''',r'''const char* a=R"tag(//keep /*keep*/ " quote \)tag"; /*remove*/ int b;''',"int a/**/b; //remove\nchar c='\\'';",'#define S(x) #x\nconst char* t=S(a/**/b);\n']
  for source in cases:
   result,_=strip_cpp(source)
   self.assertEqual(source.count('\n'),result.count('\n'))
   if ZIG.exists():self.assertEqual(preprocessed_tokens(source),preprocessed_tokens(result))
  self.assertIn('a b',strip_cpp('a/**/b')[0])
 def test_phase2_created_delimiters_fail_closed(self):
  for source in ['int a; /\\\n/ hidden\nint b;','int a; /\\\n* hidden */ int b;','int a; /*hidden*\\\n/ int b;']:
   with self.subTest(source=source),self.assertRaisesRegex(ValueError,'phase2-created'):strip_cpp(source)
 def test_preprocessor_multiline_and_line_builtin(self):
  source='#define F 3 // ignored\\\nignored too\n#define G a /* multi\nline */ b\nint x=F;\nint where=__LINE__;\n'
  result,stats=strip_cpp(source)
  self.assertEqual(source,result);self.assertEqual(stats['preserved_additional_comments'],2)
  if ZIG.exists():self.assertEqual(preprocessed_tokens(source),preprocessed_tokens(result))
 def test_every_author_and_license_group_preserved(self):
  for marker in ['Copyright 2026 X','SPDX-License-Identifier: MIT','NOTICE','bet4_claude: item = ...','alygg77','vjeux','keell0renz','matvey','Nathen','chen','Zhimao','SGLang','Marlin','Bohdan','Matsvei','author','ported from X','https://example.test']:
   source='// '+marker+'\n// whole adjacent block\nint x; // ordinary removable\n'
   result,_=strip_cpp(source)
   self.assertIn('// '+marker+'\n// whole adjacent block',result)
   self.assertNotIn('ordinary removable',result)
 def test_python_ast_docstrings_legal_blocks_and_unicode(self):
  source='# okc: problem=test\n# NOTICE\n# author X\n\n"""module doc"""\n# ordinary\ndef f():\n    """function doc"""\n    return "# string, not comment" # remove\n'
  out,stats=python_comments(source)
  self.assertEqual(ast.dump(ast.parse(source)),ast.dump(ast.parse(out)))
  self.assertIn('# NOTICE\n# author X',out);self.assertNotIn('# ordinary',out);self.assertGreater(stats['docstring_bytes'],0)
  source='\u03a9 = r\'\'\'#include <x>\n/*'+('padding '*150)+'*/\nint a; // remove\n\'\'\'\n'
  result,_=process(source);self.assertIn('\u03a9 = r',result);self.assertNotIn('// remove',result)
 def test_runtime_dispatch_marker_and_literal_anchor(self):
  native='namespace n {\n'+('int unused;\n'*100)+'void f(){switch(x){\n//@MOE1_DISPATCH@\n}}\n//custom anchor\n}\n'
  source='_MOE1_HOST_SRC = r'+repr(native)+'\ndef source():\n    anchor="//custom anchor"\n    assert _MOE1_HOST_SRC.count(anchor)==1\n    return _MOE1_HOST_SRC.replace("//@MOE1_DISPATCH@", "case 10: launch_main<51>(); break;").replace(anchor, "int injected;")\n'
  # Use a literal renderer rather than escaped repr for the actual payload.
  source=source.replace('r'+repr(native),"r'''"+native+"'''")
  out,proof=process(source);old={};new={};exec(source,old);exec(out,new)
  self.assertIn('case 10: launch_main<51>(); break;',new['source']())
  self.assertIn('int injected;',new['source']())
  self.assertEqual(strip_cpp(old['source']())[0],strip_cpp(new['source']())[0])
  self.assertEqual(set(proof['preserved_runtime_comment_anchors']),{'//@MOE1_DISPATCH@','//custom anchor'})

if __name__=='__main__':unittest.main()
