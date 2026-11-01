"""Run production dispatcher facade against functions parsed from actual packed files."""
import argparse,ast,importlib.util,sys,unittest
from pathlib import Path
from unittest.mock import patch

def run(root,directory):
 sys.path.insert(0,str(root/'tools'))
 import test_dl_verify_hybrid as original
 names=['test_all_masks_route_all_twelve_cells','test_outside_owned_domain_uses_incumbent','test_raw_abi_metadata_mismatch_uses_incumbent','test_preparation_is_per_call_and_default_skips_owned','test_unsupported_weights_preserve_incumbent_fallback','test_deadline_also_covers_unsupported_weight_preflight']
 total=0
 for mask in (4,15):
  source=(directory/f'hybrid-mask{mask}.py').read_text(encoding='utf8');tree=ast.parse(source)
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='_verify_hybrid_module']
  body=ast.unparse(ast.Module(nodes,[]));suite=unittest.TestSuite(original.VerifyHybrid(name) for name in names)
  with patch.object(original,'BODY',body):result=unittest.TextTestRunner(verbosity=2).run(suite)
  assert result.wasSuccessful(),mask;total+=result.testsRun
 print('PACKED_WRAPPER_PASS',total,'facade tests; actual packed function bodies, native arithmetic mocked')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('directory',type=Path);a=p.parse_args();run(a.root,a.directory)
