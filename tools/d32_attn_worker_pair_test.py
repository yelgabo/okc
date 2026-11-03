"""Verify the actual state-local selector wrapper without GPU execution."""
import ast
import unittest

from d32_attn_worker_pair import CANDIDATE, SUPPORT, build


class PairBinding(unittest.TestCase):
    def test_control_treatment_and_immutable_binding(self):
        marker = object()
        env = {'_d32_attention_module': lambda cfg, state: marker if cfg['supported'] else None}
        exec(SUPPORT, env)
        bind, select = env['_d32_pair_bind'], env['_d32_attention_module']
        off, on = bind({}, False), bind({}, True)
        self.assertIsNone(select({'supported': True}, off))
        self.assertIs(select({'supported': True}, on), marker)
        self.assertIsNone(select({'supported': False}, on))
        self.assertIs(select({'supported': True}, {}), marker)
        with self.assertRaises(RuntimeError):
            bind(on, False)
        on['_d32_pair_arm'] = False
        with self.assertRaises(RuntimeError):
            select({'supported': True}, on)
        with self.assertRaises(RuntimeError):
            bind({}, 1)

    def test_source_pin_and_python_entry(self):
        source = CANDIDATE.read_text()
        self.assertEqual(build(source), source + '\n' + SUPPORT)
        ast.parse(build(source))
        with self.assertRaises(ValueError):
            build(source + '\n')


if __name__ == '__main__':
    unittest.main()
