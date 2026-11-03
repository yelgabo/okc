"""Reuse the scoped attention package for exact identity-rescale elimination."""
import argparse
import json
import re
from pathlib import Path

import d32_attn_worker_candidate as scaffold


def change(cuda):
    start=cuda.index('    const float m_new = fmaxf(m_r[0], mx);',cuda.index('  uint32_t qa[4][2];'))
    end=cuda.index('#pragma unroll\n      for (int ks',start)
    old=cuda[start:end]
    before='''    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
'''
    after='''#ifndef D32_IDENTITY_SKIP
#define D32_IDENTITY_SKIP 1
#endif
    const float m_new = fmaxf(m_r[0], mx);
    const bool d32_active = m_new != -INFINITY;
    float alpha = 1.f;
    if (d32_active) alpha = ex2(m_r[0] - m_new);
    bool d32_rescale = true;
    if constexpr (D32_IDENTITY_SKIP && FLAT && WS && !ONE)
      d32_rescale = __any_sync(0xffffffffu, d32_active && alpha != 1.f);
    if (d32_active) {
'''
    new=scaffold.once(old,before,after)
    products='''#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }'''
    new=scaffold.once(new,products,'''        if (d32_rescale) {
'''+products+'''
        }''')
    return old,new


def build(disabled=False):
    base=scaffold.BASE.read_text()
    saved_change,saved_host=scaffold.native_change,scaffold.HOST
    try:
        scaffold.native_change=change
        scaffold.HOST=scaffold.HOST.replace('muse_d32_attn_score_lookahead_sm90a_v1',
            'muse_d32_identity_'+('disabled' if disabled else 'enabled')+'_sm90a_v1')
        source,control,treatment,_=scaffold.generate(base)
    finally:
        scaffold.native_change,scaffold.HOST=saved_change,saved_host
    anchor='  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.P = (int)P;'
    guard='''  TORCH_CHECK(B >= 17 && B <= 32 && L == 1 && window == 0 && P == 1 && Hq == 16 && Hkv == 2 && rope == 0,
              "D32 identity module metadata scope");
  TORCH_CHECK(n_splits == 264 && flat == 1 && nc == 1 && cs == 0 && ncl == 1 && tma == (1024 << 2),
              "D32 identity module launch scope");
  TORCH_CHECK(scale_log2 > 0. && scale_log2 <= 3.4028234663852886e38,
              "D32 identity module scale scope");
'''
    selected=scaffold.once(treatment,anchor,guard+anchor)
    if disabled:selected=scaffold.once(selected,'#define D32_IDENTITY_SKIP 1','#define D32_IDENTITY_SKIP 0')
    old="ATT7_D32_CUDA_SHA256 = "+repr(scaffold.sha(treatment))
    replacements='\nATT7_D32_CUDA_SRC = ATT7_D32_CUDA_SRC.replace('+repr(anchor)+','+repr(guard+anchor)+')\n'
    if disabled:
        replacements+="ATT7_D32_CUDA_SRC = ATT7_D32_CUDA_SRC.replace('#define D32_IDENTITY_SKIP 1','#define D32_IDENTITY_SKIP 0')\n"
    replacements+='ATT7_D32_CUDA_SHA256 = '+repr(scaffold.sha(selected))
    source=scaffold.once(source,old,replacements)
    assert len(source.encode())<1_000_000
    import ast
    ast.parse(source)
    return source,control,selected,dict(base_sha256=scaffold.EXPECTED,
        sha256=scaffold.sha(source),native_sha256=scaffold.sha(selected),disabled=disabled,
        bytes=len(source.encode()),status='SOURCE_AND_LOCAL_GATES_OPEN')


def audit(out):
    manifests=[json.loads((out/(arm+'-compile')/'manifest.json').read_text()) for arm in ('treatment','disabled')]
    incumbent=json.loads((scaffold.ROOT/'docs/steering/decode32/evidence/D32-ATTN-01/control-compile/manifest.json').read_text())
    for arm,manifest in zip(('treatment','disabled'),manifests):
        if manifest['status']!='COMPILED_NOT_TARGET_QUALIFIED':raise ValueError('compile did not finish')
        native=build(arm=='disabled')[2]
        if scaffold.sha(native)!=manifest['source_sha256']:raise ValueError('stale compiled source')
        if scaffold.sha((out/(arm+'-compile')/'device.sass').read_text())!=manifest['files']['device.sass']['sha256']:
            raise ValueError('SASS identity changed')
        if list(manifest['resources'].values())!=list(incumbent['resources'].values()):raise ValueError('resource drift')
        if manifest['layout_witness']['values']!=incumbent['layout_witness']['values']:raise ValueError('layout drift')
    if manifests[1]['files']['device.sass']['sha256']!=incumbent['files']['device.sass']['sha256']:
        raise ValueError('disabled twin differs from incumbent binary')
    sass=(out/'treatment-compile/device.sass').read_text()
    vote=sass.index('VOTE.ANY P0, P0')
    branch=re.search(r'/\*([0-9a-f]+)\*/\s+@!P0 BRA 0x([0-9a-f]+)',sass[vote:])
    if branch is None:raise ValueError('no branch after vote')
    if re.search(r'(?:FSETP|ISETP|PLOP3|PSETP)[^;]*\bP0,',sass[vote+len('VOTE.ANY P0, P0'):vote+branch.start()]):
        raise ValueError('vote predicate overwritten before branch')
    region=sass[vote+branch.end():]
    region=region[:region.index('/*'+branch[2]+'*/')]
    products=re.findall(r'FMUL (R\d+), \1, R76(?:\.reuse)? ;',region)
    if len(products)!=16:raise ValueError('branch does not skip sixteen original products')
    return dict(status='LOCAL_CODEGEN_GATE_PASS_NOT_GPU_QUALIFIED',
                registers=128,stack_bytes=0,spill_bytes=0,shared_bytes=61440,
                disabled_sass_equals_incumbent=True,products_skipped=len(products),
                vote='0x5f10',branch='0x'+branch[1],target='0x'+branch[2],
                treatment_sass_sha256=manifests[0]['files']['device.sass']['sha256'],
                disabled_sass_sha256=manifests[1]['files']['device.sass']['sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.audit:
        print(json.dumps(audit(args.out),indent=2))
        raise SystemExit
    records=[]
    for disabled in (False,True):
        source,control,native,manifest=build(disabled)
        arm='disabled' if disabled else 'treatment'
        (args.out/(arm+'.py')).write_text(source)
        (args.out/(arm+'.cu')).write_text(native)
        (args.out/'control.cu').write_text(control)
        (args.out/(arm+'.manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
        records.append(manifest)
    print(json.dumps(records,indent=2))
