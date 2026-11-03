from pathlib import Path
import re,json,collections
import argparse
p=argparse.ArgumentParser(description='Actual selected PTX final-region source/load census; no GPU');p.add_argument('cache',type=Path);a=p.parse_args();r=a.cache;results=[]
for key in ('tiny-control','tiny-relaxed','tiny2-control','tiny2-relaxed'):
 src=(r/key/'input.cu').read_text().splitlines();offset=1 # disclosed forward declaration adapter adds one source line
 rows=[];loc=''
 for no,line in enumerate((r/key/'device.ptx').read_text().splitlines(),1):
  if line.lstrip().startswith('.loc'):loc=line.strip()
  if re.search(r'\bld(?:\.[A-Za-z0-9_]+)*\.global(?:\.[A-Za-z0-9_]+)*\s',line):
   nums=[int(x) for x in re.findall(r'(?:^\.loc\s+1\s+|inlined_at 1 )([0-9]+)',loc)]
   if any(2400<=n-offset<=2585 for n in nums):
    rows.append(dict(ptx_line=no,location=loc,instruction=line.strip(),source=[dict(line=n-offset,text=src[n-offset-1]) for n in nums if 0<n-offset<=len(src)]))
 c=collections.Counter(re.search(r'\b(ld\.[^\s]+)',x['instruction'])[1] for x in rows)
 nc=[x for x in rows if '.nc.' in x['instruction']]
 print(key,dict(c),'NC',json.dumps(nc))
 results.append(dict(build=key,loads=rows,opcode_counts=dict(c),nc=nc))
(r/'post-counter-loads.json').write_text(json.dumps(results,indent=2)+'\n')
