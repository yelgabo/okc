import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from pr_host_build_contract import contract, standard


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/steering/evidence/2026-11-02-worker-pr-target'


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = (EVIDENCE / 'generated/candidate.py').read_text()
        cls.host = (ROOT / 'docs/steering/evidence/2026-11-02-worker-pr-graph-owner-relaxed/generated/host.cpp').read_text()
        cls.log = (EVIDENCE / 'target/setup-build-error.txt').read_text()

    def test_actual_failed_payload_rejected(self):
        result = contract(self.candidate, self.host, self.log)
        self.assertFalse(result['target_language_contract_pass'])
        self.assertEqual(result['candidate_effective_standard'], 17)
        self.assertEqual(result['target_minimum_standard'], 20)

    def test_later_flag_wins(self):
        self.assertEqual(standard(['-std=c++20', '-O2', '-std=c++17']), 17)
        self.assertEqual(standard(['-std=c++17', '-std=c++20']), 20)

    def test_in_memory_standard_change_is_language_only(self):
        changed = self.candidate.replace('"-std=c++17"', '"-std=c++20"')
        result = contract(changed, self.host, self.log)
        self.assertTrue(result['target_language_contract_pass'])
        self.assertFalse(result['target_environment_reproduced'])
        self.assertFalse(result['target_abi_proved'])

    def test_host_identity_required(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            contract(self.candidate, self.host + ' ', self.log)

    def test_header_evidence_required(self):
        with self.assertRaisesRegex(ValueError, 'requirement'):
            contract(self.candidate, self.host, 'no header diagnostic')

    def test_compiler_invocation_required(self):
        with self.assertRaisesRegex(ValueError, 'invocation'):
            contract(self.candidate, self.host, 'C++20 or later compatible compiler is required')

    def test_unknown_standard_rejected(self):
        with self.assertRaises(ValueError):
            standard(['-std=c++latest'])

    def test_actual_cli_stops_before_compiler(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run([sys.executable, str(ROOT / 'tools/pr_owner_host_syntax.py'),
                str(ROOT / 'docs/steering/evidence/2026-11-02-worker-pr-graph-owner-relaxed/generated/host.cpp'),
                output, '--candidate', str(EVIDENCE / 'generated/candidate.py'),
                '--target-build-log', str(EVIDENCE / 'target/setup-build-error.txt')],
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(json.loads(result.stdout)['compiler_started'])
            self.assertTrue((Path(output) / 'language-contract.json').is_file())
            self.assertFalse((Path(output) / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
