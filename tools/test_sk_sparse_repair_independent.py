"""Independent scalar oracle for SK ring B-fragment exact exponent repair.

This deliberately derives row/K ownership from mma B operand layout, not RD tables.
No candidate kernel, preparation, or launch is supplied here.
"""
import unittest

def locate(row,col,R,NT,WSK,UPW):
 cta,r=divmod(row,R);tile,g=divmod(r,8);wr,nt=divmod(tile,NT)
 unit,k=divmod(col,64);st,u=divmod(unit,WSK*UPW);wk,uu=divmod(u,UPW)
 j,k=divmod(k,16);h,k=divmod(k,8);t,half=divmod(k,2)
 return cta,st,wr*WSK+wk,uu,nt,4*g+t,2*j+h,half

def unlocate(address,R,NT,WSK,UPW):
 cta,st,warp,uu,nt,lane,reg,half=address
 wr,wk=divmod(warp,WSK);g,t=divmod(lane,4);j,h=divmod(reg,2)
 return cta*R+(wr*NT+nt)*8+g,64*(st*WSK*UPW+wk*UPW+uu)+16*j+8*h+2*t+half

def clipped(bits,base):
 e=(bits>>8)&127;off=min(7,max(0,e-base));return (bits&0x80ff)|((base+off)<<8)

def patch_word(word,half,xor7):
 assert half in (0,1) and 0<=xor7<128
 return word^(xor7<<(8+16*half))

class IndependentSKRepair(unittest.TestCase):
 def test_all_BF16_patterns_and_windows(self):
  # Includes signed zeros, subnormals, infinities and every NaN payload as BIT patterns.
  for base in range(121):
   for bits in range(65536):
    d=clipped(bits,base);x=((bits^d)>>8)&127
    self.assertEqual(d^(x<<8),bits)
    self.assertEqual(d&0x80ff,bits&0x80ff)
 def test_geometry_bijection_and_inverse(self):
  # Different row groups, splitK, UPW and NT; ragged metadata is a subset of these maps.
  for R,NT,WSK,UPW in ((24,3,4,4),(24,3,4,2),(32,4,4,1),(32,2,2,2),(32,2,2,4),(64,2,2,4)):
   seen=set()
   for r in range(2*R):
    for k in range(64*WSK*UPW*2):
     a=locate(r,k,R,NT,WSK,UPW)
     self.assertNotIn(a,seen);seen.add(a)
     self.assertEqual(unlocate(a,R,NT,WSK,UPW),(r,k))
 def test_sparse_record_and_header_bit_domains(self):
  seen=set()
  for lane in range(32):
   for reg in range(8):
    for half in (0,1):
     for xor7 in range(128):
      v=lane|(reg<<5)|(half<<8)|(xor7<<9)
      self.assertLess(v,65536)
      self.assertNotIn(v,seen);seen.add(v)
      self.assertEqual((v&31,(v>>5)&7,(v>>8)&1,v>>9),(lane,reg,half,xor7))
  self.assertEqual(len(seen),65536)
  for count in (0,1,511,512):
   for start in (0,1,(1<<22)-1):
    h=(start<<10)|count
    self.assertLess(h,1<<32)
    self.assertEqual((h>>10,h&1023),(start,count))
 def test_pair_half_repair_isolated(self):
  for half in (0,1):
   for x in range(128):
    for low,hi in ((0,65535),(0x8000,0x7f81),(0x1234,0xabcd)):
     w=low|(hi<<16);got=patch_word(w,half,x)
     self.assertEqual((got>>(16*(1-half)))&65535,(w>>(16*(1-half)))&65535)
     self.assertEqual(got&0x80ff80ff,w&0x80ff80ff)
if __name__=='__main__':unittest.main()
