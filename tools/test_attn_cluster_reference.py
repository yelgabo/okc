"""CPU Torch independent math/layout proof for queued cluster reference."""
import ast
import unittest
from attn_make_cluster_merge import FRESH


class ReferenceLayout(unittest.TestCase):
    def test_vector_reference(self):
        import torch
        def rms(x):return x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)
        gen=torch.Generator().manual_seed(311179)
        keys=torch.randn(2055,2,64,generator=gen).to(torch.float8_e4m3fn).float()
        values=torch.randn(2055,2,64,generator=gen).to(torch.float8_e4m3fn).float()
        gate=torch.randn(1,8,1024,generator=gen).to(torch.bfloat16)
        kpos=torch.arange(-2047,8);qpos=torch.arange(8)
        mask=(kpos[None,:]<=qpos[:,None])&(kpos[None,:]>=qpos[:,None]-2047)
        node=next(n for n in ast.walk(ast.parse(FRESH)) if isinstance(n,ast.FunctionDef) and n.name=='reference')
        env={'torch':torch,'keys':keys,'values':values,'gate':gate,'mask':mask,'_ob_rms':rms}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'vector_ref','exec'),env)
        q=torch.randn(8,2,8,64,generator=gen)
        for dtype in (torch.bfloat16,torch.float8_e4m3fn):
            query=q.to(dtype).float()
            actual=env['reference'](query)
            result=torch.empty(8,2,8,64)
            for j in range(8):
                for h in range(2):
                    keep=mask[j]
                    score=query[j,h] @ keys[keep,h].T * .125
                    result[j,h]=score.softmax(-1) @ values[keep,h]
            expected=(rms(result.view(1,8,16,64))*gate.float().sigmoid().view(1,8,16,64)).to(torch.bfloat16).view(1,8,1024)
            rel=(actual.float()-expected.float()).norm()/expected.float().norm()
            self.assertLess(float(rel),1e-3)
            print('CM_CPU_REF',dtype,'relative',float(rel))


if __name__=='__main__':unittest.main()
