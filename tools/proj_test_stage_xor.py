"""Actual emitted stage pack/source contracts; no GPU execution."""
import ast,unittest
from pathlib import Path
import torch

ART=Path(__file__).resolve().parents[1]/'.build/today-proj/stage-xor.py'

class StageXOR(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.source=ART.read_text(encoding='utf8');tree=ast.parse(cls.source)
  names={'bf16_bits','tile_values','window_base','code_bits','comp_units','stage_stream','balance_perm','slices_balanced','escape_table','pack_layer','xor_pack','_xor_pack_bits','sxs_repack','sxs_pack','xor_expected_layout','sxs_expected_layout'}
  ns={'torch':torch,'MAX_ESCAPE_DENOM':200,'BALANCE_GROUP':4}
  for n in tree.body:
   if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('_XOR_CONFIG','_SXS_CONFIG','_SXS_CUDA'):
    ns[n.targets[0].id]=ast.literal_eval(n.value)
  exec(compile(ast.Module([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],[]),'actual-stage-functions','exec'),ns)
  cls.ns=ns
 def compare(self,name,dense=False,empty=False):
  ns=self.ns;g=ns['_XOR_CONFIG'][name]['geom'];n=1024 if name=='slice8F' else g['R']*2;k=g['UPS']*128
  ids=torch.arange(n*k).reshape(n,k)
  bits=((63+(ids%7))<<8)|(ids&255)|((ids%2)<<15)
  if dense:bits=ids%65536
  elif empty:bits.fill_(0x3f80)
  else:bits.view(-1)[::509]=1;bits.view(-1)[250::509]=0x8000
  w=bits.to(torch.int16).view(torch.bfloat16)
  old=ns['_xor_pack_bits'](name,bits,63,torch.arange(n),n,0) if dense else ns['xor_pack'](name,w,1)
  new=ns['sxs_repack'](old,name)
  u=g['NW']*g['UPW']*g['NT'];pay=u*768;hs=(u*4+15)&~15
  a=old['stream'].reshape(-1,pay+hs);b=new['stream'].reshape(a.shape[0],pay+hs+new['stage_record_bytes'])
  self.assertTrue(torch.equal(a[:,:pay],b[:,:pay]))
  self.assertTrue(torch.equal(old['table'][old['pmax']:],new['table']))
  ah=a[:,pay:pay+u*4].contiguous().view(torch.int32).long()&0xffffffff
  bh=b[:,pay:pay+u*4].contiguous().view(torch.int32).long()&0xffffffff
  counts=ah&1023;self.assertTrue(torch.equal(counts,bh&1023))
  self.assertEqual(new['pmax'],int(counts.sum(1).max()))
  records=old['table'][:old['pmax']].view(torch.int16)
  arena=b[:,pay+hs:].contiguous().view(torch.int16)
  seen=0
  for stage in range(a.shape[0]):
   cursor=0
   for unit in range(u):
    count=int(counts[stage,unit]);start=int(ah[stage,unit]>>10);local=int(bh[stage,unit]>>10)
    self.assertEqual(local,cursor);cursor+=count;seen+=count
    self.assertTrue(torch.equal(records[start:start+count],arena[stage,local:local+count]))
   self.assertTrue(bool((arena[stage,cursor:]==0).all()))
  self.assertEqual(seen,new['records'])
  layout=ns['sxs_expected_layout'](name,2,new['pmax'])
  self.assertEqual(layout['weight_bytes'],b.shape[1]);self.assertEqual(layout['record_bytes'],arena.shape[1]*2)
  self.assertEqual(layout['stage_stride']%1024,0)
  if empty:self.assertEqual(new['pmax'],0)
 def test_all_ten_actual_streams(self):
  for name in self.ns['_XOR_CONFIG']:
   with self.subTest(name=name):self.compare(name)
 def test_dense_transport_and_empty(self):
  self.compare('qkv8F',dense=True)
  self.compare('qkv16F',empty=True)
 def test_lifetime_and_bulk_bytes(self):
  s=self.ns['_SXS_CUDA'];repair=s.index('sxs_repair(b[nt]');a=s.index('load_a(xt, ui, a);',repair)
  release=s.index('if (!C::LATE_ARRIVE',a)
  self.assertLess(repair,a);self.assertLess(a,release)
  self.assertIn('const int WBY=C::WBY+(C::STAGE?((pmax*2+15)&~15):0)',s)
  self.assertIn('ld.shared.u16',s)
  self.assertIn('s0+XBY+C::PAYLOAD+C::HEADERS',s)
  self.assertIn('NSTAGE*(sxs_sst-C::SST)',s)
 def test_bad_header_count_fails_before_copy(self):
  ns=self.ns;name='o16F';g=ns['_XOR_CONFIG'][name]['geom']
  w=torch.ones(g['R']*2,g['UPS']*64,dtype=torch.bfloat16)
  pk=ns['xor_pack'](name,w,1);u=g['NW']*g['UPW']*g['NT'];stride=u*768+((u*4+15)&~15)
  bad=dict(pk);bad['stream']=pk['stream'].clone()
  view=bad['stream'].reshape(-1,stride);h=view[:,u*768:u*768+u*4].contiguous().view(torch.int32);h[0,0]=513
  view[:,u*768:u*768+u*4]=h.view(torch.uint8)
  with self.assertRaisesRegex(AssertionError,'SXS_UNIT_COUNT'):ns['sxs_repack'](bad,name)

if __name__=='__main__':unittest.main()
