import ast,unittest
from dl_g_literal_factor import literal,pair

class Factoring(unittest.TestCase):
 def test_visible_literal_roundtrip(self):
  for s in ('', 'line\nnext\n', 'asm("\\n");\n', "'ends quote'", '"ends quote"'):
   self.assertEqual(ast.literal_eval(literal(s)),s)
 def test_exact_and_outside_ast(self):
  shared=''.join(f'__device__ int common_{i}(int x) {{ return x+{i}; }}\n' for i in range(80))
  left='// prefix A\n'+shared+'// suffix A\n';right='// prefix B\n'+shared+'// suffix B\n'
  source='A = '+literal(left)+'\nunchanged = 17\nD = {}\nD["B"] = '+literal(right)+'\n'
  out,receipt=pair(source,'A',"D['B']",'_VISIBLE_')
  self.assertTrue(receipt['outside_AST_unchanged']);self.assertGreater(receipt['saved_source_bytes'],2000)
  env={};exec(out,env)
  self.assertEqual((env['A'],env['D']['B'],env['unchanged']),(left,right,17))
  self.assertNotIn('base64',out);self.assertNotIn('zlib',out)
  with self.assertRaises(AssertionError):pair(out,'A',"D['B']",'_VISIBLE_')
if __name__=='__main__':unittest.main()
