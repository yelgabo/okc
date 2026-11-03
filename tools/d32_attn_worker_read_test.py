"""Exercise matched report identity, coverage and selection failures."""
import base64
import copy
import json
import unittest
import zlib

from d32_attn_worker_contract import decision
from d32_attn_worker_contract_test import rows
from d32_attn_worker_read import CANDIDATE, CONTROL, PREFIX, report


def fixture():
    values = rows()
    for row in values:
        target = row['tokens'] == 32 and row['context'] == 'full'
        row['selected'] = [[False], [False], [target], [target]]
    return dict(pins=dict(candidate=CANDIDATE, control=CONTROL), official_arm='attention_treatment',
                cleanup_restored=True, math_checks=144, fresh_checks=1152,
                restores=144, transitions=96, rows=values, decision=decision(values))


def encode(value):
    return PREFIX + base64.b64encode(zlib.compress(json.dumps(value).encode())).decode()


class Reader(unittest.TestCase):
    def test_exact_valid_report(self):
        self.assertEqual(report(encode(fixture())), fixture())

    def test_invalid_identity_selection_and_duplicate(self):
        for mutation in ('pin', 'selection', 'coverage', 'decision'):
            value = copy.deepcopy(fixture())
            if mutation == 'pin': value['pins']['candidate'] = 'wrong'
            if mutation == 'selection': value['rows'][0]['selected'][0] = [True]
            if mutation == 'coverage': value['fresh_checks'] -= 1
            if mutation == 'decision': value['decision']['passed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report(encode(value))
        line = encode(fixture())
        with self.assertRaises(ValueError):
            report(line + '\n' + line)


if __name__ == '__main__':
    unittest.main()
