"""CPU/source contracts for the new owned format, independent of GPU timing."""
import ast,hashlib,importlib.util,json,unittest
from pathlib import Path

from proj_sparse_xor import generate,load_c2,NAMES,PIN,repair_bits,encode_record,RUNTIME

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'okc'


class SparseXOR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=BASE/'tools/proj_sk_xor_control.py'
        spec=importlib.util.spec_from_file_location('_sx_restore',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        cls.raw=m.restore(BASE);cls.control=load_c2(cls.raw)
        cls.source,cls.cuda,cls.manifest=generate(cls.raw)

    def runtime(self):
        tree=ast.parse(self.source);ns={}
        for n in tree.body:
            if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_XOR_CONFIG':exec(compile(ast.Module([n],[]),'sx-meta','exec'),ns)
            if isinstance(n,ast.FunctionDef) and n.name=='xor_expected_layout':exec(compile(ast.Module([n],[]),'sx-layout','exec'),ns)
        return ns

    def test_tracked_raw_pin_and_unchanged_normal_entry(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(),PIN)
        normalized=self.raw.decode().replace('\r\n','\n')
        self.assertEqual(self.manifest['source_lf_sha256'],'7f4fd58cdd940f36c8d4d355f3de85687bc7f6ab04981842ce936db76d78adc2')
        self.assertTrue(self.source.startswith(normalized))
        old=ast.parse(normalized).body;new=ast.parse(self.source).body
        self.assertEqual([ast.dump(n) for n in old],[ast.dump(n) for n in new[:len(old)]])
        old_names={n.name for n in old if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertFalse(old_names&{n.name for n in new[len(old):] if isinstance(n,ast.FunctionDef)})

    def test_exact_layout_matches_original_control_formula(self):
        ns=self.runtime()
        for name in NAMES:
            c=ns['_XOR_CONFIG'][name]['original_cfg']
            _,mt,tokr,nt,nwr,wsk,upw,nstage,comp,hl,nacc,cl,outt,fuse=c
            for pmax in (0,2,64,128):
                actual=ns['xor_expected_layout'](name,0,pmax)
                expected=self.control._sk_smem(mt,tokr,nt,nwr,wsk,upw,nstage,comp,hl,cl,fuse,pmax)
                self.assertEqual(actual['dynamic'],expected,(name,pmax))
                candidate=ns['xor_expected_layout'](name,1,pmax)
                self.assertEqual(candidate['empty_arrivals'],nwr*wsk)
                self.assertEqual(candidate['escape_readers'],0)
                self.assertEqual(candidate['weight_bytes']%16,0)
                self.assertEqual(candidate['stage_stride']%1024,0)

    def test_all_BF16_bits_and_all_windows_exact(self):
        for base in range(121):
            for bits in range(65536):
                decoded,x=repair_bits(bits,base)
                if decoded^(x<<8)!=bits:raise AssertionError((base,bits))
                if decoded&0x80ff!=bits&0x80ff:raise AssertionError('sign/low-byte drift')
        for lane in range(32):
            for reg in range(8):
                for half in (0,1):
                    r=encode_record(lane,reg,half,127)
                    self.assertEqual((r&31,(r>>5)&7,(r>>8)&1,(r>>9)&127),(lane,reg,half,127))

    def test_actual_shared_read_and_mma_ordering(self):
        # SX's only new ring access is a volatile/compiler-clobbered header load.
        self.assertIn('asm volatile("ld.shared.b32 %0, [%1];"',self.cuda)
        start=self.cuda.index('            if constexpr(C::SX) {')
        repair=self.cuda.index('sx_repair(b[nt]',start)
        load_a=self.cuda.index('            load_a(xt, ui, a);',start)
        early=self.cuda.index('            if (!C::LATE_ARRIVE',load_a)
        mma=self.cuda.index('                        mma_bf16(acc[',early)
        self.assertLess(repair,load_a);self.assertLess(load_a,early);self.assertLess(early,mma)
        self.assertIn('mbar_wait<false>(cbar',self.cuda)
        self.assertIn('if constexpr (COMP && !C::SX)',self.cuda)
        # Main readers still execute both early/late original proxy-release paths.
        original=self.control._SK_SRC
        self.assertEqual(self.cuda.count('fence_proxy_async();'),original.count('fence_proxy_async();'))
        self.assertEqual(self.cuda.count('mbar_arrive(empty0 + 8 * slot);'),original.count('mbar_arrive(empty0 + 8 * slot);'))


if __name__=='__main__':unittest.main()
