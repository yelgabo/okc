"""Notice preservation negatives and exact frozen-package real-entry checks."""
import tempfile
from pathlib import Path
import unittest

import g224_notice_pack as package
import g_expert_g224_chunk_phased as generator
from g224_reference_pack import consolidate


class NoticeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as folder:
            text,_,_=generator.generate(package.ROOT,Path(folder))
        cls.text=consolidate(text)
        cls.packed=package.pack(cls.text)

    def test_exact_notices_strings_ast_and_determinism(self):
        package.verify(self.text,self.packed)
        self.assertEqual(package.pack(self.text),self.packed)
        self.assertEqual(package.sha(self.packed),package.EXPECTED)
        self.assertEqual(len(self.packed.encode()),999492)

    def test_missing_provenance_rejected(self):
        with self.assertRaises(AssertionError):
            package.verify(self.text,self.packed.replace('Submitted by alygg77','Submitted anonymously',1))

    def test_incomplete_license_rejected(self):
        for before in ('Permission is hereby granted','THE SOFTWARE IS PROVIDED','Copyright 2026 team keell0renz'):
            with self.assertRaises(AssertionError):
                package.verify(self.text,self.packed.replace(before,'REMOVED',1))

    def test_native_string_change_rejected(self):
        with self.assertRaises(AssertionError):
            package.verify(self.text,self.packed.replace('fence.proxy.async.global','fence.proxy.async.INVALID',1))

    def test_seven_selected_actual_factory_calls_equal(self):
        self.assertEqual(package.calls(self.text),package.calls(self.packed))


if __name__=='__main__':
    unittest.main()

