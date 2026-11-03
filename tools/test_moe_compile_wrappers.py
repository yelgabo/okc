import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    'moe_compile_check.sh', 'moe_compile_fused.sh', 'moe_compile_w4.sh',
    'moe_compile_legacy.sh', 'dl_moe_compile_check.sh',
)
DISASSEMBLERS = {'moe_compile_fused.sh', 'moe_compile_w4.sh'}
SOURCE = r'''_CUDA_SRC = r"""namespace {
}  // namespace
"""
_SKINNY_BLOCK = ''
SKINNY_VARIANTS = ()
MAIN_MODES = ()
VARIANT_MAIN0 = 0
def _wgmma_code(): return ''
def _skinny_tag(*args): return ''
def _cuda_source(): return _CUDA_SRC
W4_CONFIGS = ()
_W4_SRC = 'namespace {\n}  // namespace\n'
def _w4_source(): return _W4_SRC
LEGACY_CONFIGS = ()
_LEGACY_W4_SRC = 'namespace {\n}  // namespace\n'
_LEGACY_ROUTE_SRC = 'namespace {\n}  // namespace\n'
def _legacy_w4_source(): return _LEGACY_W4_SRC
_MOE1_SRC = 'namespace {\n}  // namespace\n'
_MOE1_HOST_SRC = ''
_MOE1_VARIANTS = ()
def _moe1_source(): return ''
'''


class CompileWrappers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='okc-compile-wrapper-')
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.bin = self.work / 'bin'
        self.bin.mkdir()
        self.source = self.work / 'fixture with spaces.py'
        self.source.write_text(SOURCE)
        self.log = self.work / 'logs with spaces' / 'compile.log'
        self.sass = self.work / 'output with spaces.sass'
        self.sass.write_text('previous artifact')
        self.install('docker', '''import os, subprocess, sys
args = sys.argv[1:]
if os.environ.get('FAKE_DOCKER_STATUS'):
    print('docker unavailable', file=sys.stderr)
    sys.exit(int(os.environ['FAKE_DOCKER_STATUS']))
mount = args[args.index('-v') + 1].removesuffix(':/w')
pos = next(i for i, arg in enumerate(args) if arg in ('sh', 'nvcc'))
command = [arg.replace('/w/', mount + '/').replace('cd /w ', 'cd ' + mount + ' ') for arg in args[pos:]]
sys.exit(subprocess.run(command).returncode)
''')
        self.install('python3', '''import os, sys
if len(sys.argv) > 2 and sys.argv[1] == '-c' and 'torch.utils.cpp_extension' in sys.argv[2]:
    print('-I/fake-headers')
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        self.install('nvcc', '''import os, pathlib, sys
status = int(os.environ.get('FAKE_NVCC_STATUS', '0'))
if os.environ.get('FAKE_FAIL_FILE') and not any(arg.endswith(os.environ['FAKE_FAIL_FILE']) for arg in sys.argv):
    status = 0
if status:
    print('nvcc fatal: intentional compiler failure', file=sys.stderr)
elif not os.environ.get('FAKE_QUIET'):
    print("ptxas info : Compiling entry function 'fixture'")
if not status and '-o' in sys.argv:
    pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_text('object')
sys.exit(status)
''')
        self.install('cuobjdump', '''import os, sys
status = int(os.environ.get('FAKE_DISASM_STATUS', '0'))
print('disassembler failed' if status else 'Function : fixture', file=sys.stderr if status else sys.stdout)
sys.exit(status)
''')

    def install(self, name, body):
        path = self.bin / name
        path.write_text(f'#!{sys.executable}\n' + body)
        path.chmod(0o755)

    def run_wrapper(self, name, **env):
        command = ['/bin/sh', str(ROOT / 'tools' / name), str(self.source)]
        if name in DISASSEMBLERS:
            command.append(str(self.sass))
        return subprocess.run(command, cwd=self.work, text=True, capture_output=True,
                              env={**os.environ, 'PATH': str(self.bin) + ':' + os.environ['PATH'],
                                   'KEEP': str(self.log), **env})

    def test_compiler_failure_reaches_caller(self):
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                result = self.run_wrapper(name, FAKE_NVCC_STATUS='23')
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertIn('intentional compiler failure', result.stdout + result.stderr)
                self.assertIn('intentional compiler failure', self.log.read_text())
                self.assertEqual(self.sass.read_text(), 'previous artifact')

    def test_docker_failure_reaches_caller(self):
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                result = self.run_wrapper(name, FAKE_DOCKER_STATUS='55')
                self.assertEqual(result.returncode, 55, result.stdout + result.stderr)
                self.assertIn('docker unavailable', self.log.read_text())

    def test_disassembler_failure_reaches_caller(self):
        for name in sorted(DISASSEMBLERS):
            with self.subTest(wrapper=name):
                result = self.run_wrapper(name, FAKE_DISASM_STATUS='29')
                self.assertEqual(result.returncode, 29, result.stdout + result.stderr)
                self.assertEqual(self.sass.read_text(), 'previous artifact')
                self.assertIn('disassembler failed', self.log.read_text())

    def test_legacy_first_failure_stops_before_second_compile(self):
        result = self.run_wrapper('moe_compile_legacy.sh', FAKE_NVCC_STATUS='23', FAKE_FAIL_FILE='w4.cu')
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertNotIn('Compiling entry', self.log.read_text())

    def test_quiet_success_is_success(self):
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                result = self.run_wrapper(name, FAKE_QUIET='1')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(self.log.is_file())
                if name in DISASSEMBLERS:
                    self.assertIn('Function : fixture', self.sass.read_text())

    def test_compiler_resources_remain_visible(self):
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                result = self.run_wrapper(name)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('Compiling entry', result.stdout)
                self.assertIn('Compiling entry', self.log.read_text())


if __name__ == '__main__':
    unittest.main()
