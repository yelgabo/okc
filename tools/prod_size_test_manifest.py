"""Frozen byte/actual-builder/selected-compiler evidence contracts; CPU only."""
import json,unittest
from pathlib import Path

D=Path(__file__).resolve().parents[1]/'docs/steering/evidence/2026-11-01-production-comment-fit'

class Evidence(unittest.TestCase):
 def test_frozen_files_fit_and_runtime_factories_link(self):
  m=json.loads((D/'source-manifest.json').read_text(encoding='utf8'))
  expected={4:(907642,'3b205d408edd8a238dec686fac306b5a28ac9136b42512939ae80db7c79611ec'),15:(951539,'d3f2b31ea0991e8b7e14709913e8a55d1304fe20fc47fd369e8a5f1c0334d689')}
  actual=json.loads((D/'runtime-source-ledger.json').read_text(encoding='utf8'))
  for row in actual['packages']:
   size,sha=expected[row['mask']];self.assertEqual((row['bytes'],row['sha256']),(size,sha));self.assertTrue(row['embedded_backend_identity'])
   self.assertEqual(m['artifacts']['hybrid-mask'+str(row['mask'])+'.py'],dict(bytes=size,sha256=sha));self.assertLess(size,1000000)
  self.assertEqual([r['real_load_inline_calls'] for r in actual['records']],[13,4,5])
 def test_selected_G_compiler_link_and_no_new_resources(self):
  proof=json.loads((D/'G-actual-builder-link.json').read_text(encoding='utf8'));data=json.loads((D/'G-comparison.json').read_text(encoding='utf8'))
  self.assertEqual({r['build'] for r in proof['records']},{'main','b4','tiny','tiny2','helper'})
  for row in proof['records']:self.assertTrue(row['source_hashes_match']);self.assertTrue(row['launch_bounds_and_all_ptxas_used_resources_match'])
  self.assertEqual(sum(len(r['entries']) for r in data['records']),10)
  for row in data['records']:
   a=json.loads((D/'G-local'/(row['build']+'-original')/'manifest.json').read_text())
   b=json.loads((D/'G-local'/(row['build']+'-comment')/'manifest.json').read_text())
   self.assertEqual(a['resources'],b['resources']);self.assertEqual(a['ptx_launch_bounds'],b['ptx_launch_bounds'])
 def test_owned_device_link_and_exact_three_entries(self):
  proof=json.loads((D/'owned-local/final-marker-link.json').read_text());rows=json.loads((D/'owned-local/comparison.json').read_text())
  self.assertEqual(proof['final_backend_sha256'],'c1c6065a095efb407a15c8a1fe8f6936fc3078750b1ca7adb27447846342205e')
  self.assertTrue(all(r['exact_bytes_equal'] for r in proof['proof']))
  self.assertEqual(len(rows),3);self.assertTrue(all(r['exact_instructions_operands_equal'] for r in rows))

if __name__=='__main__':unittest.main()
