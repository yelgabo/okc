import ast
from pathlib import Path
import unittest

import torch

from dl_build_g1_qualification import build

ROOT = Path(__file__).resolve().parents[1]


class ValueBoundary(unittest.TestCase):
    def test_only_device_change_is_the_reference_boundary(self):
        source = (ROOT / 'muse_decode_layer/cand/g1b2_native_experts.py').read_text('utf8')
        clean = build(source).decode()
        revised = build(source, bf16_attention=True).decode()

        def device(text):
            return next(
                ast.literal_eval(node.value) for node in ast.parse(text).body
                if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'ATT7_CUDA_SRC'
            )

        before, after = device(clean), device(revised)
        old = 'float pr8(float x) {\n  return x;'
        new = 'float pr8(float x) {\n  return __bfloat162float(__float2bfloat16_rn(x));'
        self.assertEqual(after.replace(new, old, 1), before)
        # 4-lane, 8-lane and three loop-based finalizers all normalize
        # their scaled values after pr8. No score, query, key or probability uses it.
        uses = [line for line in after.splitlines() if 'pr8(' in line and 'float pr8(' not in line]
        self.assertEqual(sum(line.count('pr8(') for line in uses), 15)
        self.assertTrue(all('inv)' in line for line in uses))
        self.assertEqual(after.count('const float rn = rsqrtf(ss * (1.f / 64.f)'), 5)

    def test_intermediate_round_is_observable_at_the_output(self):
        gen = torch.Generator().manual_seed(4291)
        values = torch.randn((128, 64), generator=gen)
        gate = torch.randn(values.shape, generator=gen).bfloat16().float().sigmoid()

        def finish(x):
            return (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * gate).bfloat16()

        rounded = values.bfloat16().float()
        self.assertGreater(torch.count_nonzero(finish(values) != finish(rounded)).item(), 0)
        encoded = values.view(torch.int32)
        rne = ((encoded + 0x7fff + ((encoded >> 16) & 1)) >> 16) << 16
        self.assertTrue(torch.equal(rne.view(torch.float32), rounded))


if __name__ == '__main__':
    unittest.main()
