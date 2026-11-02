"""Read-only public source intake. No donor execution, device or judge calls."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

OUT=Path(__file__).resolve().parent
SCRATCH=OUT.parents[3]/'.build/worker-broad-expert-architecture'
PIN='948e6fd547e703065be51529ac1db5569256774e'


def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'okc-source-review','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=30) as r:
        return r.read(),r.status,r.url


if __name__=='__main__':
    receipts=[]
    for repo in ('vllm-project/humming','inclusionAI/humming'):
        url=f'https://api.github.com/repos/{repo}/git/trees/{PIN}?recursive=1'
        try:
            raw,status,final=get(url)
            tree=json.loads(raw); assert not tree.get('truncated')
            (OUT/'tree.json').write_bytes(raw)
            receipts.append(dict(url=url,status=status,final=final,sha256=hashlib.sha256(raw).hexdigest()))
            break
        except Exception as e:receipts.append(dict(url=url,error=str(e)))
    else:raise RuntimeError(receipts)
    paths=[x['path'] for x in tree['tree'] if x['type']=='blob']
    (OUT/'intake-paths.json').write_text(json.dumps(paths,indent=2)+'\n')
    (OUT/'intake-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    selected=[x for x in tree['tree'] if x['type']=='blob' and (x['path'].startswith('humming/') or x['path'] in ('LICENSE','README.md','tests/kernels/humming/test_moe.py','benchmarks/bench_moe.py'))]
    def fetch(x):
        path=x['path'];url=f'https://raw.githubusercontent.com/{repo}/{PIN}/{path}'
        raw,status,final=get(url)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==x['sha'],path
        dest=SCRATCH/'humming-source'/(path+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        return dict(path=path,url=url,status=status,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob=blob)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        files=list(pool.map(fetch,selected))
    (OUT/'source-manifest.json').write_text(json.dumps(dict(repo=repo,pin=PIN,files=files),indent=2)+'\n')
    print(json.dumps(dict(repo=repo,tree_entries=len(paths),fetched=len(files),bytes=sum(x['bytes'] for x in files))))
