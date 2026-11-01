import os
from pathlib import Path
import unittest
from g_native_patch_ladder import build,KEYS

ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))


class Ladder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=build(ROOT)

    def test_complete_chains_and_no_unclassified_native_residual(self):
        for key in KEYS:
            record=self.data['native_ladders'][key]
            previous=record['upstream_literal_sha256']
            for step in record['steps']:
                self.assertEqual(step['before_sha256'],previous)
                self.assertTrue(step['changed_hunks'])
                previous=step['after_sha256']
            self.assertEqual(previous,record['incumbent_literal_sha256'])
            self.assertTrue(record['exact_reconstruction'])
        self.assertEqual([len(self.data['native_ladders'][key]['steps']) for key in KEYS],[21,12])
        self.assertFalse(self.data['partial_snapshots_are_conforming'])

    def test_scope_and_gamma_consumers(self):
        config=self.data['configuration']
        for key in ('MOE_DEFINES','MOE_TINY_DEFINES','MOE_TINY2_DEFINES','MOE_MID_DEFINES'):
            self.assertEqual(config['upstream'][key],config['incumbent'][key])
        self.assertTrue(config['upstream']['MOE_HAS_PDL'])
        self.assertFalse(config['incumbent']['MOE_HAS_PDL'])
        self.assertTrue(config['incumbent']['SC6_WGS'])
        gamma=self.data['gamma']
        self.assertEqual(gamma['native_slot7_expression'],"weights['post_expert_norm'].contiguous()")
        for sites in gamma['source_sites'].values():self.assertEqual(len(sites),16)
        helper=gamma['source_sites'][KEYS[1]]
        self.assertEqual(sum(site['in_selected_short_WS_function'] for site in helper),4)
        self.assertFalse(self.data['f042']['required_repairs_removed'])
        self.assertFalse(self.data['f042']['is_original_G0_helper'])


if __name__=='__main__':unittest.main()
