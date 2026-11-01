"""CPU ownership/lifetime contracts for complete per-KV-head output clusters."""
import unittest
from collections import Counter


class HeadCluster(unittest.TestCase):
    def test_fresh_reference_bf16_boundary(self):
        import inspect
        import test_attn_cluster_reference as original
        from attn_make_headcluster import FRESH
        # Independent scalar per-head reference uses the same official BF16
        # attention-value boundary; vectorization/layout remain independently checked.
        source=inspect.getsource(original.ReferenceLayout.test_vector_reference)
        import textwrap
        source=textwrap.dedent(source).replace("env['reference'](query)","env['reference'](query,True)").replace('result.view(1,8,16,64)',
            'result.to(torch.bfloat16).float().view(1,8,16,64)')
        scope={'FRESH':FRESH,'ast':__import__('ast')}
        exec(source,scope)
        scope['test_vector_reference'](self)

    def test_fresh_arm_ownership(self):
        import ast, torch
        from attn_make_headcluster import FRESH
        function=next(n for n in ast.parse(FRESH).body if isinstance(n,ast.FunctionDef) and n.name=='_hc_fresh')
        assignment=next(n for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='armdata' for t in n.targets))
        data={k:torch.arange(8.) for k in ('qkv','k_cache','v_cache')}
        data['config']={'batch':1};env={'data':data,'functions':dict.fromkeys(('ctrl','ctrl2','hc1','hc2')),'torch':torch}
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),'fresh_owner','exec'),env)
        arms=env['armdata']
        for key in ('qkv','k_cache','v_cache'):
            self.assertEqual(len({d[key].data_ptr() for d in arms.values()}),4)
            self.assertTrue(all(d[key].data_ptr()!=data[key].data_ptr() for d in arms.values()))
        arms['ctrl']['k_cache'].zero_()
        self.assertEqual(float(arms['hc1']['k_cache'].sum()),28.)
        self.assertIn('data=armdata[name]',FRESH)
        self.assertIn("for d in armdata.values():d['qkv'].copy_(bank[i])",FRESH)

    def test_attention_bf16_boundary(self):
        import torch
        # Midpoints exercise RNE ties on both even and odd BF16 mantissas.
        x=torch.tensor([1+1/256,1+3/256,1+1/256+2**-20,
                        1+1/256-2**-20,-1-1/256,0.125],dtype=torch.float32)
        rounded=x.to(torch.bfloat16).float()
        self.assertEqual(rounded[:5].tolist(),[1.0,1.015625,1.0078125,1.0,-1.0])
        # Norm input must use the rounded value, including its squared sum.
        expected=rounded.square().sum()
        self.assertNotEqual(float(expected),float(x.square().sum()))
        from attn_headcluster_patch import MERGE
        self.assertIn('__bfloat162float(__float2bfloat16_rn(acc[e]*inv));ss+=acc[e]*acc[e]',MERGE)
    def test_keys_cache_output_partition(self):
        for seq in (0,1,7,63,64,65,2047,2048,79999,80000,1000000):
            start=max(0,seq-2047);end=seq+8;tiles=(end-start+63)//64
            writes=Counter()
            for head in range(2):
                bounds=[(start+tiles*r//16*64,min(end,start+tiles*(r+1)//16*64)) for r in range(16)]
                self.assertEqual(bounds[0][0],start);self.assertEqual(bounds[-1][1],end)
                self.assertTrue(all(bounds[r][1]==bounds[r+1][0] for r in range(15)))
                self.assertEqual(sum(b-a for a,b in bounds),end-start)
                for lo,hi in bounds:
                    for j in range(8):
                        if lo<=seq+j<hi:
                            for d in range(64):writes[j,head,d]+=1
            self.assertEqual(set(writes),{(j,h,d) for j in range(8) for h in range(2) for d in range(64)})
            self.assertEqual(set(writes.values()),{1})
        outputs=Counter()
        for head in range(2):
            for rank in range(16):
                for tid in range(32):
                    lr=rank*4+tid//8;j=lr//8;h=head*8+lr%8
                    for e in range(8):outputs[j,h,(tid%8)*8+e]+=1
        self.assertEqual(len(outputs),8*16*64);self.assertEqual(set(outputs.values()),{1})

    def test_push_slots_and_local_readers(self):
        cells=Counter();sums=Counter()
        for sender in range(16):
            for tid in range(128):
                warp,lane=divmod(tid,32);g=lane//4;c=lane%4
                for hh in range(2):
                    lr=(2*warp+hh)*8+g;owner=lr//4;slot=sender*4+lr%4
                    for pp in range(4):
                        for e in range(4):cells[owner,slot*68+c*4+pp*16+e]+=1
                    if c==0:sums[owner,slot*68+64]+=1
        self.assertEqual(len(cells),16*64*64)
        self.assertEqual(set(cells.values()),{1});self.assertEqual(set(sums.values()),{1})
        for owner in range(16):
            for t in range(32):
                lr=owner*4+t//8
                for peer in range(16):
                    for e in range(8):
                        self.assertEqual(cells[owner,(peer*4+lr%4)*68+(t%8)*8+e],1)
            self.assertEqual({offset for (rank,offset) in sums if rank==owner},{slot*68+64 for slot in range(64)})
        # Generic pushes finish at cluster publication/retirement barrier;
        # consumers thereafter read ONLY storage in their own CTA, not peers.

    def test_preparation_and_shared_bijection(self):
        for head in (0,1):
            query=Counter()
            for iteration in range(8):
                for warp in range(4):
                    mask=[]
                    for lane in range(32):
                        tid=warp*32+lane;piece=tid+iteration*128;row=piece//8
                        active=(row%16)//8==head;mask.append(active)
                        if active:query[row,piece%8]+=1
                    self.assertEqual(len(set(mask)),1) # full-mask oct_sum is warp-uniform
            self.assertEqual(len(query),8*8*8);self.assertEqual(set(query.values()),{1})
        for size in (256,):
            coords=Counter(n for tid in range(128) for n in range(tid,size,128))
            self.assertEqual(set(coords),set(range(size)));self.assertEqual(set(coords.values()),{1})
        partial=Counter();sums=Counter()
        for tid in range(128):
            warp,lane=divmod(tid,32);g=lane//4;c=lane%4
            for hh in range(2):
                row=(2*warp+hh)*8+g
                for pp in range(4):
                    for e in range(4):partial[row*68+c*4+pp*16+e]+=1
                if c==0:sums[row*68+64]+=1
        self.assertEqual(set(partial),{r*68+d for r in range(64) for d in range(64)})
        self.assertEqual(set(partial.values()),{1});self.assertEqual(set(sums.values()),{1})
        self.assertEqual(len(sums),64);self.assertLess(64*68*4,49152)


if __name__=='__main__':unittest.main()
