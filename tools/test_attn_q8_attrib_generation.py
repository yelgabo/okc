"""CPU-only regression: q8 hooks are generated from pinned pure production."""
import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from attn_make_q8_attrib import patch_probe_source


class Hooks(unittest.TestCase):
    def test_pinned_generation(self):
        pure = subprocess.check_output(['git', 'show', '6da000f:muse_attention_decode/attn_one.cu'], cwd=ROOT, text=True)
        expected = subprocess.check_output(['git', 'show', 'f2b7dfa:muse_attention_decode/attn_one.cu'], cwd=ROOT, text=True)
        self.assertEqual(patch_probe_source(pure), expected)
        with self.assertRaises(AssertionError):
            patch_probe_source(expected)
        with self.assertRaises(AssertionError):
            patch_probe_source(pure.replace('constexpr uint32_t CPB', 'constexpr uint32_t DRIFT'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tools').mkdir()
            (root / 'muse_attention_decode').mkdir()
            for name in ('attn_make_q8_attrib.py', 'attn_make_bench_ab.py', 'attn_make_bench_one.py'):
                shutil.copyfile(ROOT / 'tools' / name, root / 'tools' / name)
            submission = subprocess.check_output(['git', 'show', '6da000f:muse_attention_decode/submission.py'], cwd=ROOT, text=True)
            (root / 'muse_attention_decode/submission.py').write_text(submission)
            output = root / 'bench.py'
            subprocess.run([sys.executable, str(root / 'tools/attn_make_q8_attrib.py'), str(output)], check=True)
            generated = output.read_text()
            tree = ast.parse(generated)
            sources = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_ONE_SRC' for t in n.targets)]
            self.assertEqual([s.strip() for s in sources], [expected.strip()])
            compile(generated, str(output), 'exec')
            self.assertIn('required={"ctrl","ctrl2","read_probe","zero_probe","nocopy_probe","stamp_probe"}', generated)
            self.assertEqual((root / 'muse_attention_decode/submission.py').read_text(), submission)


if __name__ == '__main__':
    unittest.main()

