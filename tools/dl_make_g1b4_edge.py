"""Generate CHECK-ONLY G1b4 D1 O->b1 timeline/stamps; no schedule change."""
import argparse, ast, hashlib, importlib.util
from pathlib import Path
PIN='1607293eee3403109df5d37b6e2d1ee34b528c3578afa6c2c8668440afde1658'

def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)

def patch(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SL_GLUE_CUDA')
    old=ast.literal_eval(node.value);s=old
    s=once(s,'  float* tail;','  float* tail;\n  unsigned long long* edge_stamp=nullptr;')
    anchor='template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>'
    stamp='''__device__ __forceinline__ void edge_mark(GlueArgs a,int k) {
  if(a.edge_stamp && threadIdx.x==0) {
    unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");
    a.edge_stamp[blockIdx.x*5+k]=t;
  }
}
'''
    s=once(s,anchor,stamp+anchor)
    s=once(s,'  asm volatile("griddepcontrol.wait;" ::: "memory");','  edge_mark(a,0);\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n  edge_mark(a,1);')
    s=once(s,'  cluster.sync();\n  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");','  edge_mark(a,2);\n  cluster.sync();\n  edge_mark(a,3);\n  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");\n  edge_mark(a,4);')
    # Existing b1 body duplicated only to pass explicit per-state timestamp storage.
    b=s[s.index('void b1(TV branch'):];assert b.rstrip().endswith('}')
    traced=once(b,'void b1(TV branch','void b1_trace(TV stamps,TV branch')
    traced=once(traced,'  if(a.rows<1','  a.edge_stamp=(unsigned long long*)stamps.data_ptr();\n  if(stamps.size(0)!=a.rows*8 || stamps.size(1)!=5) throw std::runtime_error("edge stamp bounds");\n  if(a.rows<1')
    s+='\n'+traced
    # Exact inverse proves no original statement/barrier/math was removed.
    before_waits=old.count('griddepcontrol.wait');assert s.count('griddepcontrol.wait')==before_waits
    lines=source.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['_SL_GLUE_CUDA = '+repr(s)+'\n'];source=''.join(lines)
    source=once(source,"cuda_sources=_SL_GLUE_CUDA, functions=['b1'],","cuda_sources=_SL_GLUE_CUDA, functions=['b1','b1_trace'],")
    call='_sl_cluster_module().b1(P.view(T,8192),'
    assert source.count(call)==2
    source=source.replace(call,'_edge_b1(state,P.view(T,8192),')
    return source,s

RUNTIME=r'''
import time
FP8=torch.float8_e4m3fn
RMS_EPS=1e-5
_REF_STREAM_SEED,_REF_STREAM_MEAN=5354,0.7
_REF_STREAM_SHARED,_REF_STREAM_RANK,_REF_DRAFT_SHARE=0.9,4,0.6
_EDGE_ORIGINAL_PREPARE=prepare
_EDGE_ORIGINAL_KERNEL=kernel_fn
_EDGE_COUNTS={}
_EDGE_DECODE=_decode_attention7
_EDGE_DENSE=dense

def _decode_attention7(data,state,qkv):
    if state.get('_edge_active',False):
        counts=state.setdefault('_edge_counts',{});counts['att7']=counts.get('att7',0)+1
    return _EDGE_DECODE(data,state,qkv)

def dense(x,w,T,state=None,name=None):
    result=_EDGE_DENSE(x,w,T,state,name)
    if name=='o' and state is not None and state.get('_edge_active',False):
        counts=state.setdefault('_edge_counts',{});counts['dense_o']=counts.get('dense_o',0)+1
        assert result[1]==1 and result[0].dtype==torch.bfloat16,'O fallback/output boundary'
    return result

def _edge_ptrs(value):
    if isinstance(value,torch.Tensor):return {value.data_ptr()} if value.numel() else set()
    if isinstance(value,dict):return set().union(*(_edge_ptrs(v) for v in value.values())) if value else set()
    if isinstance(value,(tuple,list)):return set().union(*(_edge_ptrs(v) for v in value)) if value else set()
    return set()


def _edge_b1(state,*args):
    if not state.get('_edge_active',False):return _sl_cluster_module().b1(*args)
    counts=state.setdefault('_edge_counts',{})
    counts['b1']=counts.get('b1',0)+1
    assert args[0].dtype==torch.bfloat16 and args[0].shape==(1,8192),'unexpected dependent P'
    mod=_sl_cluster_module()
    if state.get('_edge_trace',False):mod.b1_trace(state['_edge_stamps'],*args)
    else:mod.b1(*args)

# Python route assertions execute during warmup/capture, never extra device work.
for _name in ('_sl_cluster_after_attention','_sl_tile_after_attention'):
    def _wrap(fn,name):
        def wrapped(data,state,o_in,gate,T):
            if not state.get('_edge_active',False):return fn(data,state,o_in,gate,T)
            counts=state.setdefault('_edge_counts',{});counts[name]=counts.get(name,0)+1
            assert T==1 and gate is None,'unexpected O input boundary'
            return fn(data,state,o_in,gate,T)
        return wrapped
    globals()[_name]=_wrap(globals()[_name],_name)

def _edge_run(data,state):
    state['_edge_counts']={}
    assert state['_fork_native_eligible'] and PDL is False,'fallback/PDL forbidden'
    state['_edge_active']=True
    try:out=_EDGE_ORIGINAL_KERNEL(data,state)
    finally:state['_edge_active']=False
    c=state['_edge_counts']
    assert c.get('att7')==1 and c.get('dense_o')==1 and c.get('b1')==1 and sum(c.get(n,0) for n in ('_sl_cluster_after_attention','_sl_tile_after_attention'))==1,('inactive O->b1 edge',c)
    return out

def _edge_ns(ev):
    if hasattr(ev,'start_ns'):return ev.start_ns(),ev.end_ns()
    return round(ev.start_us()*1000),round((ev.start_us()+ev.duration_us())*1000)

def _edge_capture(data,state):
    saved=(data['stream'].clone(),data['attn_input'].clone())
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1]);_edge_run(data,state)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1])
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=_edge_run(data,state)
    graph.replay();torch.cuda.synchronize()
    return graph,out,saved

def prepare(weights):
    import random,statistics
    from torch.profiler import profile,ProfilerActivity
    from torch.autograd import DeviceType
    started=time.monotonic();deadline=started+280
    states=[_EDGE_ORIGINAL_PREPARE(weights),_EDGE_ORIGINAL_PREPARE(weights)]
    for i,s in enumerate(states):
        assert s['_fork_native_eligible'],'native eligibility required'
        s['_edge_trace']=bool(i);s['_edge_stamps']=torch.zeros((8,5),device='cuda',dtype=torch.int64)
    assert states[0]['attention_scratch'] is not states[1]['attention_scratch'],'private attention owners'
    policy=[{'mm':repr(sorted(s.get('mm_pick',{}).items(),key=str)),
             'moe':repr(sorted(s.get('moe_pick',{}).items(),key=str)),
             'bfp':repr(sorted((s.get('bfp') or {}).get('choice',{}).items(),key=str)),
             'graft':repr(sorted(s.get('compose_graft_enabled',{}).items(),key=str))} for s in states]
    assert policy[0]==policy[1],('paired dispatch differs',policy)
    flush=torch.empty(256<<20,device='cuda',dtype=torch.uint8)
    report={'schema':1,'pins':_EDGE_PINS,'schedule_changed':False,'pdl':False,'rows':[]}
    for layer in ('sliding','full'):
        for seed in (911031,911032,911033):
            assert time.monotonic()<deadline,'incomplete edge trace'
            base=_ab_inputs(1,1,layer,seed)
            data=[{k:v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in base.items()} for _ in states]
            for k in ('stream','attn_input','k_cache','v_cache'):
                assert data[0][k].data_ptr()!=data[1][k].data_ptr(),'shared paired input'
            captured=[_edge_capture(d,s) for d,s in zip(data,states)]
            for key in ('attention_scratch','ma_scratch'):
                assert not (_edge_ptrs(states[0][key]) & _edge_ptrs(states[1][key])),('shared scratch',key)
            for j in range(3):
                assert torch.equal(captured[0][1][j],captured[1][1][j]),('instrumentation output drift',layer,seed,j)
            times=[[],[]];traces=[[],[]];stamps=[]
            for repeat in range(15):
                order=[0,1];random.Random(seed+repeat).shuffle(order)
                for i in order:
                    g,out,saved=captured[i];d=data[i]
                    d['stream'].copy_(saved[0]);d['attn_input'].copy_(saved[1]);flush.zero_()
                    a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                    a.record();g.replay();z.record();z.synchronize();times[i].append(a.elapsed_time(z)*1000)
            for repeat in range(3):
                for i in (repeat%2,1-repeat%2):
                    g,out,saved=captured[i];d=data[i]
                    d['stream'].copy_(saved[0]);d['attn_input'].copy_(saved[1]);flush.zero_();torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        g.replay();torch.cuda.synchronize()
                    events=[]
                    for ev in prof.profiler.kineto_results.events():
                        if ev.device_type()==DeviceType.CUDA:
                            a,z=_edge_ns(ev);events.append((a,z,ev.device_resource_id(),ev.name()))
                    trace=summarize(events);trace['edge']=edge_summary(events);traces[i].append(trace)
                    if i==1:
                        raw=states[i]['_edge_stamps'].cpu().tolist()
                        assert all(all(x>0 for x in row) and row==sorted(row) for row in raw),'missing/nonmonotonic CTA stamps'
                        stamps.append([[round((x-row[0])/1000,4) for x in row] for row in raw])
            for i in range(2):
                report['rows'].append({'arm':('control','trace')[i],'seed':seed,'layer':layer,'cold_us':times[i],
                    'trace':sorted(traces[i],key=lambda t:t['span_us'])[1],
                    'profile_spans_us':[t['span_us'] for t in traces[i]],'stamps_relative_us':stamps if i else [],
                    'dispatch':policy[i],'path_counts':states[i]['_edge_counts']})
            assert time.monotonic()<deadline,'incomplete paired edge trace'
    assert len(report['rows'])==12
    print('G1B4_D1_EDGE_V1 '+encode_report(report),flush=True)
    return states[0]
'''

def edge_summary(events):
    """DAG-local edge coordinates; durations are not additive graph savings."""
    events=sorted(events);origin=events[0][0]
    b=[e for e in events if 'glue_cluster_boundary' in e[3] and 'false, false, true, false' in e[3]]
    if len(b)!=1:raise ValueError('expected unique D1 b1 kernel')
    consumer=b[0];prior=[e for e in events if e[2]==consumer[2] and e[1]<=consumer[0]]
    if not prior:raise ValueError('missing same-stream producer')
    producer=max(prior,key=lambda e:e[1])
    if 'sk2_kernel' not in producer[3] and 'gem' not in producer[3].lower():raise ValueError('unexpected O producer '+producer[3])
    return {'producer':producer[3],'consumer':consumer[3],'producer_end_us':(producer[1]-origin)/1000,
        'consumer_start_us':(consumer[0]-origin)/1000,'edge_gap_us':(consumer[0]-producer[1])/1000,
        'consumer_duration_us':(consumer[1]-consumer[0])/1000,
        'graph_tail_after_consumer_us':(max(e[1] for e in events)-consumer[1])/1000}

def build(candidate,root):
    import inspect
    transformed,device=patch(candidate)
    spec=importlib.util.spec_from_file_location('timeline',root/'tools/dl_make_layer_timeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.ROOT=root
    codec=(root/'tools/dl_timeline_codec.py').read_text(encoding='utf8')
    pins={'candidate_sha256':PIN,'source_main':'1a78018','sync_changed':False,'timestamps':'globaltimer 5tags/CTA; perturbed observations, not GPU counters'}
    helpers=m.extract_helpers();tree=ast.parse(helpers);inputs=next(ast.get_source_segment(helpers,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ab_inputs')
    result=transformed+'\n'+codec+'\n'+inputs+'\n'+inspect.getsource(edge_summary)+'\n_EDGE_PINS='+repr(pins)+'\n'+RUNTIME
    ast.parse(result);assert len(result.encode())<1_000_000
    return result,device

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    s,d=build(a.candidate.read_text(encoding='utf8'),a.root);a.output.write_bytes(s.encode());a.output.with_suffix('.cu').write_bytes(d.encode());print(len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
