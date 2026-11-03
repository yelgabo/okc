"""Attested NVRTC-only layout witness; actual generated CUDA stays unchanged."""
import hashlib
import re

OFFSET_ASSERTIONS = (
    'static_assert(G224_OFFSETOF(Smem, wc) == G224_OFFSETOF(Smem, wcp), "G224 alias offset");',
    'static_assert(G224_OFFSETOF(Smem, wcpB) >= G224_OFFSETOF(Smem, wc) + sizeof(((Smem*)0)->wc), "G224 separate prefix B");',
)
PROBES = dict(wc=('G224_OFFSETOF(Smem, wc)',1248), wcp=('G224_OFFSETOF(Smem, wcp)',1248),
              wcpB=('G224_OFFSETOF(Smem, wcpB)',2272), sizeofwc=('sizeof(((Smem*)0)->wc)',1024),
              sizeofsmem=('sizeof(Smem)',23096))


def adapt_device(device):
    original=device
    for assertion in OFFSET_ASSERTIONS:
        assert device.count(assertion)==1, assertion
        device=device.replace(assertion,'')
    anchor='}  // namespace okc'
    assert device.count(anchor)==1
    functions='\n'.join('extern "C" __device__ __noinline__ unsigned long long g224_probe_'+name+'() { unsigned long long result; asm volatile("mov.u64 %0, %1;" : "=l"(result) : "n"('+value[0]+')); return result; }' for name,value in PROBES.items())
    functions += '\nextern "C" __global__ void g224_layout_anchor(unsigned long long* output) { ' + ' '.join('output['+str(index)+']=g224_probe_'+name+'();' for index,name in enumerate(PROBES)) + ' }'
    device=device.replace(anchor,functions+'\n'+anchor)
    inverse=device.replace(functions+'\n'+anchor,anchor)
    assert inverse==original.replace(OFFSET_ASSERTIONS[0],'').replace(OFFSET_ASSERTIONS[1],'')
    return device


def inspect_ptx(ptx):
    results={}
    for name, (_, expected) in PROBES.items():
        match=re.search(r'(?:\.visible\s+)?\.func\b[^{}]*?\bg224_probe_'+name+r'\s*\([^{}]*\)\s*\{([^{}]*)\}',ptx,re.S)
        assert match is not None, ('missing emitted layout probe',name)
        operations=[line.strip() for line in match[1].splitlines() if line.strip() and not line.lstrip().startswith(('.', '//'))]
        assert len(operations)==3, (name,'unexpected emitted body',operations)
        literal=re.fullmatch(r'mov\.u64\s+(%rd\d+),\s*(\d+);',operations[0])
        assert literal is not None, (name,operations)
        assert re.fullmatch(r'st\.param\.b64\s+\[[^]]+\],\s*'+re.escape(literal[1])+r';',operations[1]), (name,operations)
        assert operations[2]=='ret;', (name,operations)
        value=int(literal[2]); assert value==expected, (name,value,expected)
        results[name]=dict(value=value,expected=expected,body=match[1],ptx_body_sha256=hashlib.sha256(match[1].encode()).hexdigest())
    assert results['wc']['value']==results['wcp']['value']
    assert results['wcpB']['value']>=results['wc']['value']+results['sizeofwc']['value']
    return results


def adapt_driver(driver):
    anchor='    declaration_adapter = None'
    assert driver.count(anchor)==1
    driver=driver.replace(anchor,'    device = adapt_device(device)\n'+anchor)
    anchor='    for name in expressions: check(lib.nvrtcAddNameExpression(program, name))'
    assert driver.count(anchor)==1
    driver=driver.replace(anchor,'    for name in expressions + [b"&okc::g224_layout_anchor"]: check(lib.nvrtcAddNameExpression(program, name))')
    anchor="    ptxas = cache / 'nvidia/cu13/bin/ptxas'"
    assert driver.count(anchor)==1
    driver=driver.replace(anchor,"    manifest['layout_probe_results'] = inspect_ptx(ptx.value.decode())\n"+anchor)
    anchor='    manifest.update(lowered=lowered,'
    assert driver.count(anchor)==1
    driver=driver.replace(anchor,"    manifest['layout_anchor_resources'] = entries.pop('g224_layout_anchor')\n"+anchor)
    return driver
