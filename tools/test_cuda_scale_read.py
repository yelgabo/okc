import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.cuda_scale_read import read


class ReceiptIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.job = '3aa3885f6b734f8c'
        self.payload = b'private source'
        (self.folder / 'paired.py').write_bytes(self.payload)
        self.write('paired.manifest.json', {})
        self.write('accepted.json', dict(job_id=self.job, source_sha256=hashlib.sha256(self.payload).hexdigest()))

    def write(self, name, value):
        (self.folder / name).write_text(json.dumps(value))

    def test_wrong_job_is_rejected_before_opening_its_log(self):
        self.write('status-wait.raw.json', dict(id='0000000000000000', board='muse_moe_w4a16', mode='check'))
        with self.assertRaisesRegex(ValueError, 'Owning private MoE receipt mismatch'):
            read(self.folder, self.job)

    def test_changed_payload_is_rejected(self):
        self.write('status-wait.raw.json', {})
        (self.folder / 'paired.py').write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'Accepted source/job mismatch'):
            read(self.folder, self.job)

    def test_wrong_problem_is_rejected_before_opening_its_log(self):
        self.write('status-wait.raw.json', dict(id=self.job, board='muse_decode_layer', mode='check'))
        with self.assertRaisesRegex(ValueError, 'Owning private MoE receipt mismatch'):
            read(self.folder, self.job)


if __name__ == '__main__':
    unittest.main()
