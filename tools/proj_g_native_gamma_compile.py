"""Fixed five-key G native gamma compiler gate, CPU-only. Stop on OFF/resource drift."""
import argparse,collections,hashlib,json,re,subprocess,types
from pathlib import Path
from prod_size_compile_g import bindings,sections


def norm_ptx(s):
    lines=[];depth=0;debug=False
    for line in s.splitlines():
        if not debug and re.match(r'\s*\.section\s+\.debug_',line):debug=True;depth=0
        if debug:
            depth+=line.count('{')-line.count('}')
            if depth==0 and '}' in line:debug=False
            continue
        if line.lstrip().startswith(('.file ','.file\t','.loc ','.loc\t','//')):continue
        if line.strip():lines.append(line.strip())
    return '\n'.join(lines)


def run(root,frozen,out):
    files={'baseline':root/'muse_decode_layer/cand/g1b4_stage_release.py','off':frozen/'off.py','on':frozen/'on.py'}
    pins={'baseline':'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab','off':'e17901d562b35f127c49110d222205fab254d078ae37e941a687774973a44c18','on':'d6756bb50a0362af9c9cd03d44a88c80fa593628254ffb2a931a643fccafdcad'}
    env={}
    for arm,path in files.items():
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==pins[arm],(arm,'source pin');env[arm]=bindings(raw.decode())
    driver=(root/'tools/local_cuda128_compile.py').read_text(encoding='utf8')
    old="'-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1',\n               '-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_PRMTSC=1', '-DOKC_SC6W=' + str(args.sc6), *args.define"
    assert driver.count(old)==1;driver=driver.replace(old,'*args.define')
    ns={'__name__':'_fixed_g_gamma_compile'};exec(compile(driver,'<existing-driver-actual-five-key-flags>','exec'),ns)
    flags={}
    for key,constant in [('tiny','MOE_TINY_DEFINES'),('tiny2','MOE_TINY2_DEFINES'),('mid','MOE_MID_DEFINES')]:
        f=[x for x in env['baseline'][constant] if not x.startswith('@')]
        if key=='tiny' and '-DOKC_FINPF=1' not in f:f.append('-DOKC_FINPF=1')
        flags[key]=f
    flags['b4']=['-DOKC_NW=16','-DOKC_FAST1=0','-DOKC_EPI4=0','-DOKC_WGS=1','-DOKC_B4FIN=1','-DOKC_SC6W=1']
    flags['main']=['-DOKC_NW=16','-DOKC_FAST1=0','-DOKC_EPI4=0','-DOKC_WGS=1','-DOKC_SC6W=1']
    out.mkdir(parents=True,exist_ok=True);records=[]
    result=dict(source_pins=pins,records=records,GPU=False,status='running',scope='Five fixed build keys, both bool entries; device-only, no host link/launch/admission execution')
    def save():(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        for key,f in flags.items():
            row=dict(build=key,arms={});records.append(row);save()
            for arm in files:
                cuda=env[arm]['MOE_CUDA_SRC'] if key in ('main','b4') else env[arm]['_MA_ALT_SRCS']['moe_helper0']
                folder=out/(key+'-'+arm);folder.mkdir(exist_ok=True);source=folder/'input.cu';source.write_bytes(cuda.encode())
                actual=['-DOKC_NS=3','-DOKC_NSF=3','-DOKC_FNST=4','-DOKC_PRMTSC=1','-DOKC_DQ7=1']+f+(['-DOKC_FORK_GAMMA_EFFECTIVE=1']if arm=='on'else[])
                a=types.SimpleNamespace(source=source,out=folder,cache=Path('/root/.cache/okc-cuda128'),sc6=1,define=actual,forward_declare_wgs=True)
                ns['compile_source'](a)
                sass=subprocess.run([str(a.cache/'cuda_nvdisasm-linux-x86_64-12.8.90-archive/bin/nvdisasm'),str(folder/'device.cubin')],capture_output=True,text=True,check=True).stdout
                (folder/'device.sass').write_text(sass)
                m=json.loads((folder/'manifest.json').read_text());ops=sections(folder/'device.sass');ptx=norm_ptx((folder/'device.ptx').read_text())
                row['arms'][arm]=dict(manifest=m,text_functions=len(ops),sass=ops,normalized_ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest())
                if arm=='off':
                    base=row['arms']['baseline']
                    row['off_equivalence']=dict(ptx=base['normalized_ptx_sha256']==row['arms'][arm]['normalized_ptx_sha256'],sass=base['sass']==ops,resources=base['manifest']['resources']==m['resources'],bounds=base['manifest']['ptx_launch_bounds']==m['ptx_launch_bounds'])
                    save();assert all(row['off_equivalence'].values()),(key,'OFF_EQUIVALENCE_BLOCKER',row['off_equivalence'])
                if arm=='on':
                    base=row['arms']['baseline'];r=base['manifest']['resources'];q=m['resources'];assert set(r)==set(q)
                    row['on_resources_no_worse']=all(q[n][k]<=r[n][k] for n in r for k in ('registers','stack_bytes','spill_store_bytes','spill_load_bytes')) and all(q[n]['barriers']==r[n]['barriers']for n in r) and base['manifest']['ptx_launch_bounds']==m['ptx_launch_bounds']
                    save();assert row['on_resources_no_worse'],(key,'ON_RESOURCE_BLOCKER',r,q)
                save()
            print('GAMMA_BUILD_COMPLETE',key,flush=True)
        result['status']='compiled_all_fixed_keys';save()
    except BaseException as error:
        result.update(status='blocked',blocker=str(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('frozen',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.root,a.frozen,a.out)
