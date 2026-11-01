"""Independent CPU interpretation of actual emitted SX packer/stream.
Set SK_SPARSE_ARTIFACT to the exact prototype source under review; no CUDA executes.
"""
import ast,os,unittest,hashlib
from pathlib import Path
import torch
from test_sk_sparse_repair_independent import unlocate

class ActualPacker(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if 'SK_SPARSE_ARTIFACT' not in os.environ:raise unittest.SkipTest('Set SK_SPARSE_ARTIFACT to the exact emitted prototype under review')
  cls.path=Path(os.environ['SK_SPARSE_ARTIFACT']);s=cls.path.read_text(encoding='utf8');t=ast.parse(s)
  names={'bf16_bits','tile_values','window_base','code_bits','comp_units','stage_stream','balance_perm','slices_balanced','escape_table','pack_layer','xor_pack','_xor_pack_bits'}
  nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in names]
  ns={'torch':torch,'MAX_ESCAPE_DENOM':200,'BALANCE_GROUP':4}
  for n in t.body:
   if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_XOR_CONFIG':ns['_XOR_CONFIG']=ast.literal_eval(n.value)
  exec(compile(ast.Module(nodes,type_ignores=[]),'<actual-SX-packer-functions>','exec'),ns)
  cls.ns=ns;print('INDEPENDENT_SX_PACKER_SHA',hashlib.sha256(s.encode()).hexdigest())
 def check_name(self,name,dense=False):
  g=self.ns['_XOR_CONFIG'][name]['geom'];R=g['R'];K=64*g['UPS']*2;N=1024 if name=='slice8F' else R*2
  ids=torch.arange(N*K,dtype=torch.int64).reshape(N,K)
  bits=((63+(ids%7))<<8)|(ids&255)|((ids%2)<<15)
  # Sparse extrema in many lanes/registers/halves; <=.5% admission cap.
  bits.view(-1)[::509]=0x0001;bits.view(-1)[250::509]=0x8000
  if dense:bits=ids%65536
  w=bits.to(torch.int16).view(torch.bfloat16)
  pk=(self.ns['_xor_pack_bits'](name,bits,63,torch.arange(N),N,0) if dense else self.ns['xor_pack'](name,w,1));self.assertIsNotNone(pk)
  ctrl=None if dense else self.ns['xor_pack'](name,w,0)
  if not dense:self.assertIsNotNone(ctrl)
  n=pk['n'];table=pk['table'];rowmap=table[pk['pmax']:].to(torch.int64)
  self.assertEqual(rowmap.numel(),n);self.assertEqual(len(set(rowmap.tolist())),n)
  # Same row map as actual pinned control, including original qkv balancing.
  if ctrl is not None:
   ct=ctrl['table'];self.assertTrue(torch.equal(rowmap,ct[:,2*ctrl['pmax']:].reshape(-1).long()))
  rec=table[:pk['pmax']].view(torch.int16).to(torch.int64)&65535
  nw,upw,nt,wsk=g['NW'],g['UPW'],g['NT'],g['WSK'];pay=nw*upw*nt*768;hs=(nw*upw*nt*4+15)&~15
  stream=pk['stream'].reshape(2,n//R,pay+hs);base=pk['base4']&255
  visited=set();used=[];maxcount=0
  for st in range(2):
   for cta in range(n//R):
    h=stream[st,cta,pay:].contiguous().view(torch.int32).long()&0xffffffff
    for warp in range(nw):
     for uu in range(upw):
      for ti in range(nt):
       idx=(warp*upw+uu)*nt+ti;hdr=int(h[idx]);count=hdr&1023;start=hdr>>10
       maxcount=max(maxcount,count)
       self.assertLessEqual(count,512);self.assertLessEqual(start+count,len(rec))
       unit=stream[st,cta,idx*768:(idx+1)*768]
       lo=unit[:512].reshape(32,16).long();c=unit[512:].reshape(32,8).long();raw=[]
       for v in range(16):
        q=v//8;b=v%8;code=(c[:,q*4+b%4]>>(4*(b//4)))&15
        raw.append(lo[:,v]|(((base+(code&7))|((code&8)<<4))<<8))
       restored=torch.stack(raw,1)
       for i in range(start,start+count):
        r=int(rec[i]);lane=r&31;reg=(r>>5)&7;half=(r>>8)&1;x=r>>9
        key=(cta,st,warp,uu,ti,lane,reg,half)
        self.assertNotIn(key,visited);visited.add(key);used.append(i)
        restored[lane,2*reg+half]^=x<<8
       for lane in range(32):
        for v in range(16):
         row,col=unlocate((cta,st,warp,uu,ti,lane,v//2,v%2),R,nt,wsk,upw)
         self.assertEqual(int(restored[lane,v]),int(bits[int(rowmap[row]),col]))
  self.assertEqual(sorted(used),list(range(pk['records'])))
  if dense:self.assertEqual(maxcount,512)
 def test_actual_all_offered_geometry_packs(self):
  for name in self.ns['_XOR_CONFIG']:
   with self.subTest(name=name):self.check_name(name)
 def test_actual_dense_all_bit_patterns_transport(self):
  self.check_name('qkv8F',dense=True)
 def test_empty_and_nonfinite_rejection(self):
  for name in ('qkv16F','o16F'):
   g=self.ns['_XOR_CONFIG'][name]['geom'];w=torch.ones(g['R']*2,g['UPS']*64,dtype=torch.bfloat16)
   for variant in (0,1):self.assertIsNotNone(self.ns['xor_pack'](name,w,variant))
   pk=self.ns['xor_pack'](name,w,1);self.assertEqual(pk['records'],0);self.assertEqual(pk['pmax'],0)
   w[0,0]=float('nan')
   for variant in (0,1):self.assertIsNone(self.ns['xor_pack'](name,w,variant))
if __name__=='__main__':unittest.main()
