"""CPU source/actual derived-coefficient gates only; no Triton/GPU execution."""
import ast, copy, os, unittest
from pathlib import Path
from proj_g_preexpert_gamma import build, census, CONSUMERS, DERIVED, FIELD

ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1].parent/'okc'))
SOURCE=(ROOT/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8')


class Tests(unittest.TestCase):
    def test_default_off_and_pin(self):
        s,m=build(SOURCE)
        self.assertEqual(s,SOURCE);self.assertFalse(m['enabled'])
        with self.assertRaisesRegex(AssertionError,'unreviewed'):build(SOURCE+'\n',True)
    def test_independent_ast_transform_and_complete_callers(self):
        out,m=build(SOURCE,True)
        class Expected(ast.NodeTransformer):
            def visit_BinOp(self,n):
                n=self.generic_visit(n)
                if isinstance(n.op,ast.Add) and isinstance(n.left,ast.Constant) and n.left.value==1.0 and any(isinstance(x,ast.Name) and x.id=='GPRE' for x in ast.walk(n.right)):
                    return n.right
                return n
            def visit_Subscript(self,n):
                n=self.generic_visit(n)
                if isinstance(n.value,ast.Name) and n.value.id=='state' and isinstance(n.slice,ast.Constant) and n.slice.value=='g_preexp':n.slice=ast.Constant(FIELD)
                return n
            def visit_Assign(self,n):
                n=self.generic_visit(n)
                if ast.unparse(n.targets[0])=="st['g_preexp']":return [n,ast.parse(DERIVED.strip()).body[0]]
                return n
        expected=Expected().visit(ast.parse(SOURCE))
        self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(ast.parse(out),include_attributes=False))
        self.assertEqual((m['census']['load_count'],m['census']['caller_count']),(8,15))
        tree=ast.parse(out)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in CONSUMERS:
                self.assertFalse(any(isinstance(x,ast.BinOp) and isinstance(x.op,ast.Add) and isinstance(x.left,ast.Constant) and x.left.value==1.0 and any(isinstance(v,ast.Name) and v.id=='GPRE' for v in ast.walk(x.right))for x in ast.walk(node)))
        self.assertIn('st["g_preexp"] = weights["pre_expert_norm"].contiguous()',out)
        self.assertTrue(m['source_reverse_proof'])
    def test_actual_prepare_expression_all_finite_bf16(self):
        import torch
        out,_=build(SOURCE,True)
        node=next(n for n in ast.walk(ast.parse(out)) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==f"st['{FIELD}']")
        pattern=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        raw=pattern[torch.isfinite(pattern)].clone();saved=raw.view(torch.int16).clone()
        st={'g_preexp':raw}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-generated-prepare-coefficient>','exec'),{'st':st,'torch':torch})
        # Literal baseline precompute_effective_weight operation, in BF16 parameter dtype.
        baseline=(raw+1.0).detach()
        self.assertEqual(baseline.dtype,torch.bfloat16)
        self.assertTrue(torch.equal(st[FIELD].view(torch.int16),baseline.view(torch.int16)))
        self.assertTrue(torch.equal(raw.view(torch.int16),saved));self.assertIs(st['g_preexp'],raw)
        self.assertNotEqual(st[FIELD].data_ptr(),raw.data_ptr());self.assertTrue(st[FIELD].is_contiguous())
        w=torch.tensor([2**-8],dtype=torch.bfloat16)
        self.assertNotEqual(float((w.float()+1)[0]),float((w+1)[0])) # intentional inherited coefficient difference
        print('ACTUAL_FINITE_BF16_COEFFICIENTS',len(raw))


if __name__=='__main__':unittest.main()
