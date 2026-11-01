"""CPU Torch executes the transformed E4M3 integer operations; no CUDA needed."""
import ast
import types
import unittest
from dl_fork_attention import E4M3_SOURCE


class RNEFixtures(unittest.TestCase):
    def test_torch_float8_boundaries(self):
        import torch
        class LowerTo(ast.NodeTransformer):
            def visit_Call(self,node):
                node=self.generic_visit(node)
                if isinstance(node.func,ast.Attribute) and node.func.attr=='to':
                    return ast.copy_location(ast.Call(func=ast.Name(id='_convert',ctx=ast.Load()),args=[node.func.value,*node.args],keywords=node.keywords),node)
                return node
        def convert(x,dtype,bitcast=False):
            if bitcast:
                assert dtype=='u32'
                return x.contiguous().view(torch.int32).to(torch.int64) & 0xffffffff
            return x.to(dtype)
        tl=types.SimpleNamespace(uint32='u32',int32=torch.int64,float32=torch.float32,abs=torch.abs,where=torch.where)
        tree=ast.fix_missing_locations(LowerTo().visit(ast.parse(E4M3_SOURCE)))
        env={'tl':tl,'_convert':convert};exec(compile(tree,'cpu_e4m3_source','exec'),env)
        codes=torch.arange(127,dtype=torch.uint8)
        vals=codes.view(torch.float8_e4m3fn).float()
        mids=(vals[:-1]+vals[1:])/2
        neighbors=torch.cat((mids,torch.nextafter(mids,torch.full_like(mids,float('inf'))),torch.nextafter(mids,torch.full_like(mids,-float('inf')))))
        g=torch.Generator().manual_seed(41511)
        random=torch.randn(10000,generator=g)*torch.pow(2.,torch.randint(-15,12,(10000,),generator=g).float())
        edge=torch.tensor([0.,-0.,448.,449.,464.,465.,float('inf'),-float('inf'),float('nan'),-float('nan')])
        x=torch.cat((vals,-vals,neighbors,-neighbors,random,edge)).float()
        actual=env['_f32_to_e4m3'](x).to(torch.uint8)
        expected=x.to(torch.float8_e4m3fn).view(torch.uint8)
        self.assertTrue(torch.equal(actual,expected),str([(float(x[i]),int(actual[i]),int(expected[i])) for i in torch.nonzero(actual!=expected).flatten()[:8]]))
        print('CPU_TORCH_E4M3_FIXTURES',x.numel(),'all byte-exact; subnormal/normal ties, signedzero, overflow, Inf, NaN')


if __name__=='__main__':unittest.main()
