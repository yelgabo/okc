"""Independent integer coverage proof, no external kernel import or GPU work."""
import unittest

def coordinates(gm,gn,broken=False):
 return [(p%(gn if broken else gm),p//gm) for p in range(gm*gn)]

class LocalityMap(unittest.TestCase):
 def test_corrected_rectangular_bijection(self):
  for gm in range(1,33):
   for gn in range(1,33):
    actual=coordinates(gm,gn)
    expected={(m,n) for m in range(gm) for n in range(gn)}
    self.assertEqual(len(actual),len(set(actual)))
    self.assertEqual(set(actual),expected)
 def test_square_can_hide_defect(self):
  for g in range(1,33):self.assertEqual(coordinates(g,g),coordinates(g,g,True))
 def test_rectangles_reject_original(self):
  for gm,gn in ((2,3),(3,2),(1,8),(8,1),(8,448)):
   self.assertNotEqual(set(coordinates(gm,gn,True)),set(coordinates(gm,gn)))
  self.assertEqual(sum(m<8 for m,n in coordinates(8,448,True)),64)
 def test_padded_tail_and_expert_ownership(self):
  # Five valid padded M tiles, eight allocated tiles; each valid tile's expert
  # and each N tile must survive exactly once, with no allocation-tail work.
  experts=[0,0,2,2,3];gm,gn=8,7
  active=[(m,n,experts[m]) for m,n in coordinates(gm,gn) if m<len(experts)]
  self.assertEqual(len(active),35)
  self.assertEqual(set(active),{(m,n,experts[m]) for m in range(5) for n in range(gn)})
if __name__=='__main__':unittest.main()
