"""Independent actual global-to-stage record conversion proof; no GPU execution.
Set SK_SPARSE_ARTIFACT to the stage prototype (pinned SHA printed in transcript).
"""
import ast,unittest
import torch
import test_sk_sparse_repair_packer_cpu as _global

class StageRecords(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  _global.ActualPacker.setUpClass();cls.ns=dict(_global.ActualPacker.ns);cls.path=_global.ActualPacker.path
  src=cls.path.read_text(encoding='utf8');tree=ast.parse(src)
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sxs_repack')
  exec(compile(ast.Module([fn],[]),'<actual-stage-repacker>','exec'),cls.ns)
  cls.cuda=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SXS_CUDA')
  cls.global_cuda=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_XOR_CUDA')
 def check_conversion(self,name,kind):
  g=self.ns['_XOR_CONFIG'][name]['geom'];N=g['R']*2;K=g['UPS']*64*7
  ids=torch.arange(N*K).reshape(N,K);bits=((63+ids%7)<<8)|(ids&255)|((ids%2)<<15)
  if kind=='sparse':bits.reshape(-1)[::509]=1;bits.reshape(-1)[250::509]=0x8000
  elif kind=='zero':bits.fill_(0x3f80)
  elif kind=='dense':bits=ids%65536
  else:
   bits.fill_(0x3f80);bits[:8,:64]=1
  perm=torch.arange(N).reshape(-1,4).flip(0).reshape(-1)
  old=self.ns['_xor_pack_bits'](name,bits,63,perm,N,0)
  oldstream=old['stream'].clone();oldtable=old['table'].clone()
  stage=self.ns['sxs_repack'](old,name)
  self.assertTrue(torch.equal(oldstream,old['stream']));self.assertTrue(torch.equal(oldtable,old['table']))
  units=g['NW']*g['UPW']*g['NT'];payload=units*768;hb=(units*4+15)&~15;groups=14
  a=oldstream.reshape(groups,payload+hb);b=stage['stream'].reshape(groups,-1)
  oh=a[:,payload:payload+4*units].contiguous().view(torch.int32).long()&0xffffffff
  nh=b[:,payload:payload+4*units].contiguous().view(torch.int32).long()&0xffffffff
  counts=oh&1023;expected_cap=max(sum(int(v) for v in row) for row in counts)
  self.assertEqual(stage['pmax'],expected_cap);arena_bytes=(2*expected_cap+15)&~15
  self.assertEqual(b.shape[1],payload+hb+arena_bytes)
  self.assertTrue(torch.equal(a[:,:payload],b[:,:payload]))
  self.assertTrue(torch.equal(stage['table'],oldtable[old['pmax']:]))
  glob=oldtable[:old['pmax']].view(torch.int16)
  arena=b[:,payload+hb:].contiguous().view(torch.int16)
  used=0
  for group in range(groups):
   prefix=0
   for u in range(units):
    count=int(oh[group,u]&1023);start=int(oh[group,u]>>10)
    self.assertEqual(int(nh[group,u]&1023),count);self.assertEqual(int(nh[group,u]>>10),prefix)
    self.assertLessEqual(prefix+count,expected_cap)
    self.assertTrue(torch.equal(glob[start:start+count],arena[group,prefix:prefix+count]))
    prefix+=count;used+=count
   self.assertTrue(bool((arena[group,prefix:]==0).all()))
  self.assertEqual(used,old['records']);self.assertEqual(used,stage['records'])
  if kind=='zero':self.assertEqual(expected_cap,0)
  if kind=='oneunit':self.assertEqual(expected_cap,512)
 def test_all_geometry_sparse_empty_and_max_unit(self):
  for name in self.ns['_XOR_CONFIG']:
   for kind in ('sparse','zero','oneunit'):
    with self.subTest(name=name,kind=kind):self.check_conversion(name,kind)
 def test_dense_stage_prefix_capacity(self):self.check_conversion('qkv8F','dense')
 def test_source_transport_and_lifetime(self):
  s=self.cuda
  self.assertIn('const int WBY=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);',s)
  self.assertIn('const uint32_t stage_tx = UPS * XU + WBY;',s)
  self.assertIn('mbar_expect_tx(full0 + 8 * slot, stage_tx);',s)
  self.assertIn('static_assert(!STAGE || SX',s)
  self.assertIn('pmax<=C::NW*UPW*NT*512',s)
  begin=s.index('uint32_t sxh[NT]');repair=s.index('sxs_repair(b[nt]',begin);load=s.index('load_a(xt, ui, a)',repair);release=s.index('mbar_arrive(empty0 + 8 * slot)',load)
  self.assertLess(begin,repair);self.assertLess(repair,load);self.assertLess(load,release)
  for term in ('fence_proxy_async();','__syncwarp();','mbar_arrive(empty0 + 8 * slot);','named_bar(1, NC);'):
   self.assertEqual(s.count(term),self.global_cuda.count(term),term)
if __name__=='__main__':unittest.main()
