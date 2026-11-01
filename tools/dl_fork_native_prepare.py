"""Generate G1b2 from the measured native candidate, changing only its host preparation."""
import argparse
import ast
import hashlib
from pathlib import Path

PIN = '7935ff411d5e7ec1af41b25c7fecbff3524c4311efa65b6635b05e1d5bd41025'

HOST = '''
def _fork_build_tasks(weights, st):
    dev = weights['w13'].device
    prop = torch.cuda.get_device_properties(dev)
    eligible = bool(st['ma_sfx']) and (prop.major, prop.minor) == (9, 0)
    st['_fork_native_eligible'] = eligible
    st['_fork_build_errors'] = {}
    if not eligible:
        return []
    def build(m):
        with torch.cuda.device(dev):
            try:
                _fork_native_moe(m, st)
            except RuntimeError as exc:
                st['_fork_build_errors'][m] = str(exc)[-300:]
    # Each representative selects a different module/cache key.
    return [lambda m=m: build(m) for m in (1, 8, 16, 32, 128)]

def _fork_install_fallback(weights, st, reason):
    st['_fork_native_eligible'] = False
    st['ma_w'], st['ma_scratch'] = (), ()
    installed = _fork_install_owned_moe(weights, st)
    st['_fork_fallback_kind'] = 'owned' if installed else 'Marlin'
    if not installed:
        _fork_prepare_marlin(weights, st)
        dev = weights['w13'].device
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        for m in (1,8,16,32,33,128,256):
            x = torch.zeros((m,4096), device=dev, dtype=torch.bfloat16)
            ids = torch.full((m,8), -1, device=dev, dtype=torch.int32)
            ids[:,0] = 0
            tw = torch.full((m,8), .125, device=dev, dtype=torch.float32)
            st['_fork_marlin_adapter'].okc_run((),(),x,ids,tw,torch.empty_like(x),0,32,sms)
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    st['_fork_moe_status'] = st['_fork_fallback_kind'] + ': ' + reason
    print('FORK_MOE_ROUTE', st['_fork_moe_status'], flush=True)

def _ma_prepare(weights, st):
    # All background compilation has joined. Reject preceding CUDA execution faults.
    torch.cuda.synchronize()
    errors = st.pop('_fork_build_errors')
    if not st['_fork_native_eligible'] or errors:
        reason = 'native build failed '+str(sorted(errors.items())) if errors else 'scale/architecture fallback'
        return _fork_install_fallback(weights, st, reason)
    if set(st['ma_ext']) != {'tiny','tiny2','mid','b4','main'}:
        raise RuntimeError('unexpected native module keys: '+str(sorted(st['ma_ext'])))
    dev = weights['w13'].device
    _fork_native_ma_prepare(weights, st)
    torch.cuda.synchronize()
'''


def patch(source):
    if hashlib.sha256(source.encode()).hexdigest() != PIN:
        raise ValueError('unexpected G1b input')
    original_tree = ast.parse(source)
    start = source.index('def _ma_prepare(weights, st):')
    probe = source.index('    for m in (1,8,16,32,33,128,256):\n        x=torch.zeros', start)
    source = source[:start] + HOST + source[probe:]
    start = source.index("            st['_fork_native_eligible']=False", source.index('FORK_MOE_RESOURCE\' not in str(exc):', start))
    stop = source.index('        torch.cuda.synchronize()', start)
    source = source[:start] + "            return _fork_install_fallback(weights, st, str(exc)[-300:])\n" + source[stop:]
    line = next(l for l in source.splitlines() if l.startswith('    ths = [threading.Thread'))
    assert 'lambda: _moe(32)' in line and source.count(line) == 1
    source = source.replace(line, '''    builders = [_att7, bfp__get_ext, _graft_glue_module, _graft_b2_module, _sl_cluster_module]
    builders.extend(_fork_build_tasks(weights, st))
    ths = [threading.Thread(target=_build, args=(f,)) for f in builders]''')
    source = source.replace('# Modified by yelgabo: G1b,', '# Modified by yelgabo: G1b2, parallel native preparation,')
    source = source.replace('"G1b backend"', '"G1b2 backend"')
    updated = ast.parse(source)
    def constants(tree):
        return [ast.dump(n) for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
    assert constants(updated) == constants(original_tree)
    if len(source.encode()) >= 1_000_000:
        raise ValueError('G1b2 exceeds the judge size limit')
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    payload = patch(args.input.read_text('utf-8')).encode()
    args.output.write_bytes(payload)
    print(len(payload), hashlib.sha256(payload).hexdigest())
