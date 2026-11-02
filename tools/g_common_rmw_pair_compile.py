"""Exact paired native factory capture and selected device compile; no GPU."""
import argparse,ast,hashlib,json,subprocess,time,types
from pathlib import Path
from g_native_patch_ladder import flags
from prod_size_compile_g import bindings,sections


def capture(source):
    ns=bindings(source);ns.update(flags(source));calls=[]
    def load_inline(**kw):
        calls.append(kw)
        return types.SimpleNamespace(okc_set_pdl=lambda *a:None)
    class Wrapper:
        def __init__(self,m):self._module=m
        def __getattr__(self,k):return getattr(self._module,k)
    ns.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda:(9,0))),
              load_inline=load_inline,_RouterExt=Wrapper)
    for name in ('_ma_ext','_main_defines','_fork_native_moe'):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(ast.get_source_segment(source,node),ns)
    for on in (False,True):
        state=dict(ma_ext={},ma_sfx=('-DOKC_PRMTSC=1',),ma_sc6=True,_srm_on=on)
        for t in (1,8,16,32,128):ns['_fork_native_moe'](t,state)
    assert len(calls)==7,'five OFF plus two distinct ON builds'
    return calls


def run(root,source,out):
    text=source.read_text(encoding='utf8');calls=capture(text)
    out.mkdir(parents=True,exist_ok=True)
    ledger=[]
    for c in calls:
        ledger.append(dict(name=c['name'],cuda_sha256=hashlib.sha256(c['cuda_sources'].encode()).hexdigest(),
            cpp_sha256=hashlib.sha256(c['cpp_sources'].encode()).hexdigest(),flags=c['extra_cuda_cflags'],functions=c['functions']))
    (out/'factory-capture.json').write_text(json.dumps(ledger,indent=2)+'\n')
    driver=(root/'tools/local_cuda128_compile.py').read_text(encoding='utf8')
    old="'-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1',\n               '-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_PRMTSC=1', '-DOKC_SC6W=' + str(args.sc6), *args.define"
    assert driver.count(old)==1;driver=driver.replace(old,'*args.define')
    ns={'__name__':'_paired_selected_compile'};exec(driver,ns)
    records=[]
    for i in (0,1,5,6):
        c=calls[i];folder=out/str(i);folder.mkdir(exist_ok=True)
        path=folder/'input.cu';path.write_text(c['cuda_sources'])
        a=types.SimpleNamespace(source=path,out=folder,cache=Path('/root/.cache/okc-cuda128'),
            sc6=1,define=[f for f in c['extra_cuda_cflags'] if f.startswith('-D')],forward_declare_wgs=True)
        start=time.monotonic();ns['compile_source'](a);elapsed=time.monotonic()-start
        sass=subprocess.run([str(a.cache/'cuda_nvdisasm-linux-x86_64-12.8.90-archive/bin/nvdisasm'),str(folder/'device.cubin')],capture_output=True,text=True,check=True).stdout
        (folder/'device.sass').write_text(sass)
        m=json.loads((folder/'manifest.json').read_text());ops=sections(folder/'device.sass')
        records.append(dict(factory_index=i,seconds=elapsed,manifest=m,ops=ops))
    for off,on in ((records[0],records[2]),(records[1],records[3])):
        assert off['manifest']['resources']==on['manifest']['resources'],'resource parity'
        assert all(v['spill_store_bytes']==v['spill_load_bytes']==0 for v in on['manifest']['resources'].values()),'new spill'
    summary=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),factory_builds=7,
        extra_builds=2,selected_compile_seconds=[r['seconds'] for r in records],records=records,
        GPU=False,scope='Exact actual source/flags; both bool entries Tiny/Tiny2 OFF/ON; device-only NVRTC12.8 excludes host link/runtime13.0')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PAIRED_COMPILE_PASS',summary['source_sha256'],summary['selected_compile_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.root,a.source,a.out)
