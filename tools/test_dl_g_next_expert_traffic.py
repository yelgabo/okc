import base64
from pathlib import Path
import struct
import unittest
import zlib
from dl_g_next_expert_traffic import decode,model
class TrafficTests(unittest.TestCase):
 def test_capsule_and_payload(self):
  root=Path(__file__).resolve().parents[2]/"okc"
  d=decode((root/"muse_moe_w4a16/profiles/frontier-normalized.1c6ebc76b91d4b7f.log").read_text(encoding="utf8"))
  self.assertEqual((d["tokens"],d["local_routes"],d["active_experts"],d["jobs"],d["counts"][24]),(256,337,20,21,121))
  self.assertEqual(d["extra_bytes"],27525120)
  self.assertEqual(d["phase_payload"]["W13"]["extra_bytes"],18350080)
  self.assertEqual(d["phase_payload"]["W2"]["extra_bytes"],9175040)
 def test_group_boundaries(self):
  for count,extra in [(0,0),(1,0),(32,0),(64,0),(65,1),(128,1),(129,2),(256,3)]:
   d=model([count]+[0]*31);self.assertEqual(d["extra_jobs"],extra)
 def test_malformed_capsules(self):
  for raw in (b"WGF1",struct.pack("<4sHH",b"WGF1",0,256),struct.pack("<4sHH",b"BAD!",132,256)):
   with self.assertRaises((ValueError,struct.error)):decode("WGSFRONT_TRACE "+base64.b64encode(zlib.compress(raw)).decode())
if __name__=="__main__":unittest.main()
