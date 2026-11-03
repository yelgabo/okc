import json
from pathlib import Path
import tempfile
import unittest

from tools.check_cuda_research import digest, prose_units, sections, validate


class ResearchCoverage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.txt'
        self.source.write_text('# Research\n```python\n# code comment\n```\n## Pipeline\nBody\n')
        self.note = self.root / 'note.md'
        self.note.write_text('[source](source.txt)\n')
        self.evidence = self.root / 'kernel.cu'
        self.evidence.write_text('kernel v1\n')
        self.catalog = self.root / 'catalog.json'
        self.data = {
            'source': {'path': 'source.txt', 'sha256': digest(self.source)},
            'sections': [{**row, 'note': 'note.md'} for row in sections(self.source.read_text())],
            'notes': ['note.md'],
            'evidence': [{'path': 'kernel.cu', 'sha256': digest(self.evidence), 'notes': ['note.md']}],
        }

    def errors(self):
        self.catalog.write_text(json.dumps(self.data))
        return validate(self.root, self.catalog)[0]

    def test_complete_coverage_ignores_fenced_comments(self):
        self.assertEqual(len(self.data['sections']), 2)
        self.assertEqual(self.errors(), [])

    def test_changed_evidence_names_affected_note(self):
        self.evidence.write_text('kernel v2\n')
        self.assertIn('Evidence changed: kernel.cu; review note.md', self.errors())

    def test_omitted_section_is_rejected(self):
        self.data['sections'].pop()
        self.assertIn('Source section coverage differs from the catalog', self.errors())

    def test_changed_source_is_rejected(self):
        self.source.write_text(self.source.read_text() + 'New evidence\n')
        self.assertTrue(any('Original attachment changed' in error for error in self.errors()))

    def test_changed_claim_is_rejected(self):
        self.data['claims'] = {'note.md': {'units': prose_units(self.note.read_text())}}
        self.note.write_text('Changed claim\n')
        self.assertTrue(any('Note changed: note.md' in error for error in self.errors()))

    def test_broken_link_and_unmapped_section_are_rejected(self):
        self.note.write_text('[missing](absent.md)\n')
        self.data['sections'][0]['note'] = 'absent.md'
        self.assertEqual(len(self.errors()), 2)


if __name__ == '__main__':
    unittest.main()
