import ast
import hashlib
import os
from pathlib import Path
import subprocess
import unittest

from dl_fork_moe_empty import DEVICE_PIN, GUARD, HEAD, device_node, patch_empty_routing

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('OKC_EMPTY_BASE', ROOT / 'muse_decode_layer/cand/g1b2_native_experts.py'))


def function(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


class EmptyTests(unittest.TestCase):
    def test_exact_transform_and_pin(self):
        source = BASE.read_text(encoding='utf-8')
        before = device_node(source)[1]
        after = device_node(patch_empty_routing(source))[1]
        self.assertEqual(hashlib.sha256(before.encode()).hexdigest(), DEVICE_PIN)
        self.assertEqual(after.replace(GUARD, ''), before)
        self.assertEqual(after.count(GUARD), 1)
        with self.assertRaises(AssertionError):
            patch_empty_routing(source.replace('const int nj = S.njF + S.njS;', 'const int nj = S.njS + S.njF;'))
        with self.assertRaises(AssertionError):
            patch_empty_routing(patch_empty_routing(source))

    def test_empty_reachability_source(self):
        d = device_node(BASE.read_text(encoding='utf-8'))[1]
        ranges = function(d, '__device__ __noinline__ void wgs_ranges')
        self.assertIn('S.wcp[nj] = (int)carryA', ranges)
        self.assertIn('S.wtot = carryA', ranges)
        self.assertIn('if (lane < 4) S.wgr[lane] = res;', ranges)
        phase = function(d, '__device__ __forceinline__ void wgs_phase')
        self.assertLess(phase.index('if (gs >= ge)'), phase.index('wgs_cta_of('))
        self.assertIn('break;', phase[:phase.index('wgs_cta_of(')])
        # Exactly one call and one definition. The call is inside the nonempty
        # phase loop; its division by total cost is unreachable for zero ranges.
        self.assertEqual(d.count('wgs_cta_of('), 2)

    def test_actual_extracted_cpp(self):
        d = device_node(BASE.read_text(encoding='utf-8'))[1]
        old = function(d, HEAD.rstrip())
        new = old.replace(HEAD, HEAD + GUARD)
        common = r'''
#include <stdint.h>
#include <algorithm>
#include <iostream>
using std::min;
#define __device__
#define __forceinline__
struct Checked { int data[16]{}; int size=0; mutable int reads=0;
 operator const int*() const { ++reads; return data; }
 int operator[](int i) const { ++reads; return (i<0 || i>=size)?0:data[i]; }};
struct Smem { Checked wcp, wcpB, jC; };
int wgs_cost(int pb,int c) { return 10+c+pb; }
int wgs_unitc(int pb) { return 19+pb; }
'''
        main = r'''
int main() {
 Smem empty; int checks=0;
 for(int pb=0;pb<2;++pb) for(int G=1;G<=136;++G) for(int c=0;c<G;++c) for(int k=0;k<2;++k) {
   if(wgs_stage_at(empty,0,pb,0)!=0) return 1; ++checks;
 }
 if(empty.wcp.reads || empty.wcpB.reads || empty.jC.reads) return 2;
 old_stage_at(empty,0,0,0);
 if(empty.jC.reads==0 || empty.wcp.reads==0) return 3;
 for(int nj=1;nj<=15;++nj) for(int pb=0;pb<2;++pb) {
   Smem s; s.jC.size=nj; s.wcp.size=nj+1; s.wcpB.size=nj+1;
   for(int j=0;j<nj;++j) {s.jC.data[j]=1+j*3; int cost=(16*wgs_cost(pb,s.jC.data[j])+wgs_unitc(pb))*(pb?16:32);
     s.wcp.data[j+1]=s.wcp.data[j]+cost; s.wcpB.data[j+1]=s.wcpB.data[j]+cost;}
   int total=pb?s.wcpB.data[nj]:s.wcp.data[nj];
   for(int x=0;x<=total;x+=17) {if(wgs_stage_at(s,nj,pb,x)!=old_stage_at(s,nj,pb,x)) return 4; ++checks;}
 }
 std::cout << "EMPTY_CPP_PASS checks=" << checks << " empty_reads=0 original_metadata_reads=" << empty.jC.reads << "\n";
 return checks>100000?0:5;
}
'''
        folder = ROOT / '.build/today-moe'
        folder.mkdir(parents=True, exist_ok=True)
        cpp = folder / 'empty_host.cpp'
        cpp.write_text(common + old.replace('wgs_stage_at(', 'old_stage_at(') + '\n' + new + '\n' + main, encoding='utf-8')
        linux_root = '/mnt/c/' + str(ROOT).replace('\\', '/')[3:]
        zig = os.environ.get('OKC_EMPTY_ZIG', '/root/.cache/okc-host-tools/zig-0-15-2/ziglang/zig')
        args = ['wsl', '-d', 'Ubuntu', '--exec', zig, 'c++', '-std=c++17', '-O1',
                linux_root + '/.build/today-moe/empty_host.cpp', '-o', linux_root + '/.build/today-moe/empty_host']
        subprocess.run(args, check=True, timeout=180)
        result = subprocess.run(['wsl', '-d', 'Ubuntu', '--exec', linux_root + '/.build/today-moe/empty_host'], check=True, text=True, capture_output=True)
        print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
