import sys
sys.path.insert(0,'../okc-today-proj/tools')
import prod_size_comments as p
fixtures=[('adjacency','int/**/x;'),('raw','const char*x=R"tag(// not comment /* text */)tag"; // erase\n'),('escaped','const char*x="quote\\\"//kept"; /* erase */\n'),('splice','// erase\\\ncontinued\nint x;'),('line','int x;/* erase\nsecond */ int y;\n'),('legal','// Author: Alice\n// Copyright A\n// Apache License\nint x; // erase\n')]
for name,s in fixtures:
 out,pr=p.strip_cpp(s);assert out.count('\n')==s.count('\n')
 if name=='legal':assert '// Author: Alice' in out
 if name=='raw':assert '/* text */' in out
 if name=='adjacency':assert out=='int x;'
a='''# Author: Alice
# Copyright A
# Licensed under Apache License
x = "# not a comment" # erase
''';b,_=p.python_comments(a);assert '# Author: Alice' in b and '"# not a comment"' in b
print('7 fixture checks PASS')
