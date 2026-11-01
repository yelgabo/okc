import ast
import builtins
from pathlib import Path
import symtable
import unittest

from dl_g_production_slice import build,fold_disabled_gle,literal_inventory,reduce_module

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'muse_decode_layer/cand/g1b4_stage_release.py'
if not SOURCE.exists():SOURCE=ROOT.parent/'okc/muse_decode_layer/cand/g1b4_stage_release.py'


class ProductionSlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=SOURCE.read_text(encoding='utf8')
        cls.output,cls.manifest=build(cls.original)

    def test_eager_aliases_dynamic_globals_and_mutation_history(self):
        text='''value=1
table={}
table['a']=4
def prepare(x):return x+value+table['a']
old_prepare=prepare
value=7
def prepare(x):return old_prepare(x)+1
table['b']=8
def unused(x):return x*100
def kernel_fn(x,state):return prepare(x)+table['b']
'''
        reduced,removed=reduce_module(text)
        old={};new={};exec(text,old);exec(reduced,new)
        for x in (1,8,16,32,128,256):
            self.assertEqual(old['prepare'](x),new['prepare'](x))
            self.assertEqual(old['kernel_fn'](x,None),new['kernel_fn'](x,None))
        self.assertTrue(any(n['name']=='unused' for n in removed))
        self.assertIn('old_prepare=prepare',reduced)

    def test_every_retained_statement_unchanged_and_native_literals_exact(self):
        folded,_=fold_disabled_gle(self.original)
        originals=[ast.dump(n) for n in ast.parse(folded).body]
        cursor=0
        for n in ast.parse(self.output).body:
            value=ast.dump(n);cursor=originals.index(value,cursor)+1
        self.assertEqual(literal_inventory(self.original),literal_inventory(self.output))
        self.assertEqual(self.manifest['removed_literals'],{})
        self.assertIn('def _release_expert_self_test(',self.output)
        self.assertIn('_release_normal_prepare = prepare',self.output)
        self.assertIn('def _gle_frt(',self.output)
        self.assertNotIn('def _gle_after_attention(',self.output)

    def test_no_additional_unbound_module_globals(self):
        def missing(source):
            table=symtable.symtable(source,'G-production','exec')
            available=set(dir(builtins))|{s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
            def used(t):return {s.get_name() for s in t.get_symbols() if s.is_global()}.union(*(used(c) for c in t.get_children()))
            return set().union(*(used(c) for c in table.get_children()))-available
        self.assertLessEqual(missing(self.output),missing(self.original))

    def test_empty_flag_proof_and_negative_mutations(self):
        folded,proof=fold_disabled_gle(self.original)
        self.assertEqual(proof['value'],{})
        for changed in (self.original.replace('GLE_T = {}','GLE_T = {8: True}',1),
                        self.original+'\nG_UNKNOWN_ALIAS=GLE_T\n',
                        self.original+'\nGLE_T.clear()\n'):
            with self.assertRaises(AssertionError):fold_disabled_gle(changed)
        old=[n for n in ast.parse(self.original).body if isinstance(n,ast.FunctionDef) and n.name=='_after_attention'][-1]
        new=[n for n in ast.parse(folded).body if isinstance(n,ast.FunctionDef) and n.name=='_after_attention'][-1]
        self.assertEqual(ast.dump(old.body[-1]),ast.dump(new.body[0]))
        for tokens in (1,8,16,32,128,256):
            for gate in (None,object()):
                results=[]
                for node in (old,new):
                    env={'GLE_T':{},'_gle_prior_after':lambda *args:args}
                    exec(compile(ast.Module([node],[]),'GLE-dead-branch','exec'),env)
                    results.append(env['_after_attention']('data','state','o',gate,tokens))
                self.assertEqual(results[0],results[1])

    def test_pinned_budget_and_no_claim_of_fit(self):
        self.assertEqual(self.manifest['bytes'],854399)
        self.assertFalse(self.manifest['H200_qualified'])
        self.assertGreater(self.manifest['bytes']+330000,1000000)
        # The original license/NOTICE is between pure production and the
        # appended self-test, not necessarily a final-file trailer.
        marker=self.original.index('Apache License',self.original.index('_g1a_prepare=prepare'))
        first=self.original.rfind('\n',0,marker)+1
        end=self.original.index('def _release_private_scratch',first)
        self.assertIn(self.original[first:end],self.output)


if __name__=='__main__':unittest.main()
