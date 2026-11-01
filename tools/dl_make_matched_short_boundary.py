"""Build a check-only comparison with identical upstream state and dispatch."""
from pathlib import Path
import sys

from dl_make_layer_timeline import extract_helpers
from dl_short_rms_owner import candidate


def duplicate(source, reference, start, end, rename):
    begin = reference.index(start)
    finish = reference.index(end, begin) + len(end)
    block = reference[begin:finish]
    for before, after in rename:
        block = block.replace(before, after)
    anchor = source.index(end, source.index(start)) + len(end)
    return source[:anchor] + "\n" + block + source[anchor:]


def build(source):
    variant = candidate(source)
    symbol = [('ffn_boundary_short', 'ffn_boundary_owner'), ('boundary_kernel', 'boundary_owner_kernel')]
    source = duplicate(source, variant, '__launch_bounds__(THREADS) boundary_kernel(', '\n}',
                       symbol)
    marker = '\n__launch_bounds__(THREADS) boundary_owner_kernel'
    assert source.count(marker) == 1
    source = source.replace(marker, '\n__global__ void __cluster_dims__(CTAS, 1, 1) __launch_bounds__(THREADS) boundary_owner_kernel')
    signature = 'std::vector<torch::Tensor> ffn_boundary_short('
    source = duplicate(source, variant, signature, 'double eps_next);', symbol)
    body = variant[variant.rindex(signature):].split('\n}', 1)[0] + '\n}'
    assert source.count(body) == 1
    source = source.replace(body, body+'\n'+body.replace('ffn_boundary_short','ffn_boundary_owner').replace('boundary_kernel','boundary_owner_kernel'))
    source = source.replace('"ffn_boundary_short", "pre_norm"', '"ffn_boundary_short", "ffn_boundary_owner", "pre_norm"')
    source = duplicate(source, variant, 'def _bnd(stream', 'bnd["eps_next"])',
                       [('_bnd(', '_bnd_owner('),('ffn_boundary_short','ffn_boundary_owner')])
    fn = source[source.rindex('def kernel_fn('):]
    assert fn.count('_bnd(stream, unpooled, state)') == 1
    source += '\n'+fn.replace('def kernel_fn(', 'def _owner_layer(').replace('_bnd(stream, unpooled, state)', '_bnd_owner(stream, unpooled, state)')
    return source+'\n'+extract_helpers()+'\n'+PROBE


PROBE = r'''
_REF_STREAM_SEED, _REF_STREAM_MEAN = 5354, 0.7
_REF_STREAM_SHARED, _REF_STREAM_RANK, _REF_DRAFT_SHARE = 0.9, 4, 0.6
_matched_prepare = prepare
_timeline = lambda state: None

def prepare(weights):
    import random
    import statistics
    state = _matched_prepare(weights)
    assert state['bnd_ours']
    flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
    rng = random.Random(82117)
    rows = []
    for length in (1,8):
        for layer in ('sliding','full'):
            for seed in (720181,720182):
                base = _ab_inputs(1,length,layer,seed)
                tf32 = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    ids = _ab_ref_front(base,weights)['biased'].topk(TOP_K).indices
                    base['local_start'].fill_(32*int(torch.bincount((ids//32).flatten(),minlength=8).argmax()))
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = tf32
                for scope in ('boundary','layer'):
                    f = torch.randn((length,1024),device='cuda',dtype=torch.bfloat16)
                    arms = []
                    for changed in (False,False,True,True):
                        data = {k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in base.items()}
                        short = _bnd_owner if changed else _bnd
                        step = _owner_layer if changed else kernel_fn
                        def call():
                            return short(data['stream'],f,state) if scope=='boundary' else step(data,state)
                        call()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            out = call()
                        data['stream'].copy_(base['stream'])
                        data['attn_input'].copy_(base['attn_input'])
                        graph.replay()
                        torch.cuda.synchronize()
                        snapshot = [t.clone() for t in out]
                        if arms:
                            for actual,expected in zip(snapshot,arms[0][3]):
                                if scope=='boundary' or not actual.is_floating_point():
                                    assert torch.equal(actual,expected), (scope,length,layer,seed)
                                else:
                                    rel=(actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)
                                    assert float(rel)<0.001,(scope,length,layer,seed,float(rel))
                        arms.append((graph,data,out,snapshot))
                    samples = [[],[],[],[]]
                    for rep in range(32):
                        for idx in rng.sample(range(4),4):
                            graph,data,_,_ = arms[idx]
                            data['stream'].copy_(base['stream'])
                            data['attn_input'].copy_(base['attn_input'])
                            flush.zero_()
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record()
                            graph.replay()
                            end.record()
                            end.synchronize()
                            samples[idx].append(start.elapsed_time(end)*1000)
                    rows.append((scope,length,layer,seed,[round(statistics.median(t),3) for t in samples]))
    for row in rows:
        print('MATCHED_SHORT',*row,flush=True)
    return state
'''

if __name__ == '__main__':
    source = build(Path('muse_decode_layer/cand/layer_attn31s.py').read_text(encoding='utf-8'))
    assert len(source.encode()) < 1_000_000
    compile(source,'<matched>','exec')
    Path(sys.argv[1]).write_text(source,encoding='utf-8',newline='\n')
