import copy
import ast
import gc
import re
import types
import textwrap
import unittest

import g_pr_graph_owner as treatment
from test_g_pr_compressed_owner import Tensor, actual_entries
from test_g_timed_silu_table import capture

ORIGINAL=treatment.BASE.read_bytes().decode()
SOURCE=treatment.build(ORIGINAL)
Tensor.stride=lambda self: (self.shape[1],1)


class Record:
    def __init__(self, module, pointer):
        self.module, self.pointer = module, pointer
        self.references, self.queued = 1, False

    def retain(self):
        self.references += 1

    def release(self):
        self.references -= 1
        assert self.references >= 0
        if self.references == 0:
            assert not self.queued
            self.queued=True
            self.module.retirement.append(self)


class OwnedTensor(Tensor):
    def __init__(self, source, record):
        super().__init__(source.shape, source.dtype)
        self.record=record
        self.record.pointer=self.pointer

    def __del__(self):
        self.record.release()


class Graph:
    next_id=1

    def __init__(self, handle=9):
        self.handle, self.capture_id = handle, Graph.next_id
        Graph.next_id += 1
        self.records=[]

    def add(self, record):
        record.retain()
        self.records.append(record)

    def clone(self):
        result=Graph(self.handle+1)
        for record in self.records:
            result.add(record)
        return result

    def destroy(self):
        records, self.records = self.records, []
        for record in records:
            record.release()


class Executable(Graph):
    def __init__(self, graph):
        super().__init__()
        self.inflight, self.destroyed = False, False
        for record in graph.records:
            self.add(record)

    def destroy(self):
        self.destroyed=True
        if not self.inflight:
            super().destroy()

    def complete(self):
        self.inflight=False
        if self.destroyed:
            super().destroy()


class Owner:
    def __init__(self, module, source):
        self.module=module
        self.record=Record(module, 0)
        self.tensor=OwnedTensor(source, self.record)

    def before_read(self, stream):
        if self.module.hook_failure:
            raise RuntimeError(self.module.hook_failure)
        self.module.hooks.append((self.tensor.pointer, stream,
            self.module.graph.capture_id if self.module.graph else None))
        if self.module.graph:
            self.module.graph.add(self.record)


class Module:
    def __init__(self, failure=None):
        self.failure, self.hook_failure = failure, None
        self.graph=None
        self.hooks, self.retirement, self.allocations, self.unwind = [], [], [], []
        self.worker_events=[]

    def drain(self, fail_completion=False):
        while self.retirement:
            self.retirement.pop(0)
            self.worker_events += ['worker-context','completion']
            if fail_completion:
                self.worker_events.append('fatal-abort-no-unmap')
                return
            self.worker_events += ['unmap','release','address-free','pop','primary-release']

    def allocate(self, source, stream):
        completed=[]
        acquired=['primary']
        for stage in ('capability', 'granularity', 'reserve', 'create', 'property',
                      'map', 'access', 'copy', 'copy_completion', 'user_create'):
            if self.failure==stage:
                self.unwind=(['completion'] if 'copy' in completed or stage=='copy' else [])
                self.unwind += list(reversed(acquired))
                return None, 'no-treatment:'+stage
            completed.append(stage)
            if stage in ('reserve','create','map'):
                acquired.append(stage)
        owner=Owner(self, source)
        self.allocations.append((source.pointer, owner.tensor.pointer, stream))
        return owner, 'generic-owned'


def integrated(module=None):
    env, events, captured=actual_entries()
    lt_class=next(node for node in ast.parse(ORIGINAL).body
                  if isinstance(node,ast.ClassDef) and node.name=='_GraftLt')
    method=next(node for node in lt_class.body
                if isinstance(node,ast.FunctionDef) and node.name=='__deepcopy__')
    method_scope={}
    exec(textwrap.dedent(ast.get_source_segment(ORIGINAL,method)),method_scope)
    env['_GraftLt'].__deepcopy__=method_scope['__deepcopy__']
    def startup(torch, weights, state, *args, **kwargs):
        for tokens in (128,256):
            plan=state['graft_lt'].get(('pr',tokens))
            if plan is None:
                continue
            owner=plan.get('_pr_graph_owner')
            assert plan['weight'] is (owner.tensor if owner else state['pr'])
        events.append('original_startup')
    env['_release_expert_self_test']=startup
    suffix=SOURCE[SOURCE.index('\n_PR_GRAPH_MODULE = None\n'):]
    exec(suffix, env)
    env['_PR_GRAPH_MODULE']=module or Module()
    for name in ('_graft_lt_plan', '_graft_lt_run'):
        exec(treatment.function(SOURCE, name), env)
    return env, events, captured


def run(env, state, tokens):
    return env['_graft_lt_run'](Tensor((tokens,8192)), state['graft_lt'][('pr',tokens)])


class Contracts(unittest.TestCase):
    def test_source_reversal_and_seven_factories(self):
        self.assertTrue(treatment.validate(SOURCE, ORIGINAL)['exact_reversal'])
        before, routes_before=capture(ORIGINAL)
        after, routes_after=capture(SOURCE)
        self.assertEqual(len(before),7)
        self.assertEqual(before,after)
        self.assertEqual(routes_before,routes_after)

    def test_actual_final_prepare_and_both_plans(self):
        env, events, captured=integrated()
        state=env['prepare']({})
        self.assertEqual(events,['normal_prepare','complete_prior_work','original_startup'])
        for tokens in (128,256):
            plan=state['graft_lt'][('pr',tokens)]
            self.assertIsNot(plan['weight'],state['pr'])
            self.assertIs(plan['weight'],plan['_pr_graph_owner'].tensor)
            run(env,state,tokens)
        self.assertEqual(len(env['_PR_GRAPH_MODULE'].hooks),2)
        self.assertEqual(captured,[state['graft_lt'][('pr',128)]['weight'].pointer]*2)
        self.assertEqual(state['pr'].shape,(4608,8192))
        self.assertEqual(state['pr'].dtype,'bf16')

    def test_independent_interleaved_states_and_nonpr(self):
        env, _, _=integrated()
        first, second=env['prepare']({}),env['prepare']({})
        for state,tokens in ((first,128),(second,256),(first,256),(second,128)):
            run(env,state,tokens)
        self.assertNotEqual(first['graft_lt'][('pr',128)]['weight'].pointer,
                            second['graft_lt'][('pr',128)]['weight'].pointer)
        count=len(env['_PR_GRAPH_MODULE'].hooks)
        plan=first['graft_lt'][('o',128)]
        env['_graft_lt_run'](Tensor((128,1024)),plan)
        self.assertEqual(len(env['_PR_GRAPH_MODULE'].hooks),count)
        self.assertNotIn('_pr_graph_owner',plan)

    def test_capture_clone_abort_inflight_destroy(self):
        module=Module()
        env, _, _=integrated(module)
        state=env['prepare']({})
        module.graph=Graph()
        run(env,state,128)
        run(env,state,256)
        graph,module.graph=module.graph,None
        clone=graph.clone()
        executable=Executable(clone)
        executable.inflight=True
        graph.destroy()
        clone.destroy()
        del state
        gc.collect()
        self.assertFalse(module.retirement)
        executable.destroy()
        self.assertFalse(module.retirement)
        executable.complete()
        self.assertEqual(len(module.retirement),1)
        self.assertEqual(module.retirement[0].references,0)
        module.drain()
        self.assertEqual(module.worker_events,['worker-context','completion','unmap','release',
                                              'address-free','pop','primary-release'])

    def test_reused_handle_not_capture_cache(self):
        module=Module()
        env, _, _=integrated(module)
        state=env['prepare']({})
        first,second=Graph(9),Graph(9)
        for graph in (first,second):
            module.graph=graph
            run(env,state,128)
        self.assertNotEqual(first.capture_id,second.capture_id)
        self.assertEqual(len(first.records),1)
        self.assertEqual(len(second.records),1)
        first.destroy()
        second.destroy()

    def test_capture_hook_failures_prevent_lt_read(self):
        for failure in ('capture-info','invalidated-capture','graph-retain','read-context-mismatch'):
            with self.subTest(failure=failure):
                module=Module()
                env, _, captured=integrated(module)
                state=env['prepare']({})
                module.hook_failure=failure
                with self.assertRaisesRegex(RuntimeError,failure):
                    run(env,state,128)
                self.assertEqual(captured,[])

    def test_actual_prepare_composition_capture_hook(self):
        module=Module()
        env, _, _=integrated(module)
        state=env['prepare']({})
        observed=[]
        class CapturedGraph(Graph):
            def replay(self):
                self.replays=getattr(self,'replays',0)+1
        class CaptureScope:
            def __init__(self,graph):
                self.graph=graph
            def __enter__(self):
                module.graph=self.graph
            def __exit__(self,*args):
                module.graph=None
        def create_graph():
            graph=CapturedGraph()
            observed.append(graph)
            return graph
        event=types.SimpleNamespace(record=lambda: None,elapsed_time=lambda other: 0.001)
        env['torch'].cuda.CUDAGraph=create_graph
        env['torch'].cuda.graph=CaptureScope
        env['torch'].cuda.Event=lambda **kwargs: event
        env['torch'].ones=lambda *shape,**kwargs: Tensor(shape,'f32')
        env['torch'].sum=lambda *args,**kwargs: None
        exec(treatment.function(ORIGINAL,'_compose_times'),env)
        env['_compose_times']([lambda: run(env,state,128),lambda: run(env,state,256)],reps=2)
        self.assertEqual(len(module.hooks),4)
        self.assertEqual(sum(hook[2] is not None for hook in module.hooks),2)
        self.assertEqual([len(graph.records) for graph in observed],[1,1])
        self.assertEqual([graph.replays for graph in observed],[2,2])
        for graph in observed:
            graph.destroy()

    def test_staged_fallback_keeps_actual_original_pr(self):
        for failure in ('capability','granularity','reserve','create','property','map',
                        'access','copy','copy_completion','user_create'):
            with self.subTest(failure=failure):
                env, _, captured=integrated(Module(failure))
                state=env['prepare']({})
                self.assertEqual(state['_pr_graph_status'],'no-treatment:'+failure)
                for tokens in (128,256):
                    self.assertIs(state['graft_lt'][('pr',tokens)]['weight'],state['pr'])
                    run(env,state,tokens)
                self.assertEqual(captured,[state['pr'].pointer]*2)
                resources=[name for name in ('map','create','reserve','primary')
                           if name in env['_PR_GRAPH_MODULE'].unwind]
                self.assertEqual([name for name in env['_PR_GRAPH_MODULE'].unwind if name!='completion'],resources)

    def test_tagged_plan_deepcopy_alias_and_wrong_weight_negative(self):
        env, _, _=integrated()
        state=env['prepare']({})
        plan=state['graft_lt'][('pr',128)]
        cloned=copy.deepcopy(plan)
        self.assertIs(cloned['weight'],plan['weight'])
        self.assertIs(cloned['_pr_graph_owner'],plan['_pr_graph_owner'])
        cloned['weight']=state['pr']
        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
            env['_graft_lt_run'](Tensor((128,8192)),cloned)
        with self.assertRaisesRegex(RuntimeError,'tag mismatch'):
            copy.deepcopy(cloned)
        for field,wrong in (('shape',(1,8192)),('dtype','f16'),('device','other-device')):
            with self.subTest(field=field):
                wrong_tensor=Tensor(plan['weight'].shape,plan['weight'].dtype)
                wrong_tensor.pointer=plan['weight'].pointer
                setattr(wrong_tensor,field,wrong)
                bad=dict(plan,weight=wrong_tensor)
                with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
                    env['_graft_lt_run'](Tensor((128,8192)),bad)
        wrong_stride=Tensor(plan['weight'].shape,plan['weight'].dtype)
        wrong_stride.pointer=plan['weight'].pointer
        wrong_stride.stride=lambda: (1,4608)
        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
            env['_graft_lt_run'](Tensor((128,8192)),dict(plan,weight=wrong_stride))

    def test_abort_and_eager_retirement_exactly_once(self):
        module=Module()
        env, _, _=integrated(module)
        state=env['prepare']({})
        module.graph=Graph()
        run(env,state,256)
        module.graph.destroy()
        module.graph=None
        del state
        gc.collect()
        self.assertEqual(len(module.retirement),1)
        self.assertTrue(module.retirement[0].queued)

    def test_failed_worker_completion_never_unmaps(self):
        module=Module()
        env, _, _=integrated(module)
        state=env['prepare']({})
        run(env,state,128)
        del state
        gc.collect()
        module.drain(fail_completion=True)
        self.assertEqual(module.worker_events,['worker-context','completion','fatal-abort-no-unmap'])
        self.assertNotIn('unmap',module.worker_events)

    def test_both_pr_plan_checks_reject_no_compressed_reader(self):
        module=Module()
        env, _, captured=integrated(module)
        env['_fork_lt_no_split']=lambda *args: False
        state=env['prepare']({})
        self.assertNotIn(('pr',128),state['graft_lt'])
        self.assertNotIn(('pr',256),state['graft_lt'])
        self.assertEqual(captured,[])
        gc.collect()
        self.assertEqual(len(module.retirement),1)

    def test_callback_signal_only_and_before_read_coverage(self):
        callback=re.search(r'static void CUDA_CB graph_release.*?\n}',treatment.HOST,re.S).group()
        self.assertNotIn('cuCtx',callback)
        self.assertNotIn('cuMem',callback)
        self.assertNotIn('sleep',callback)
        self.assertIn('callback_finished.store',callback)
        run_source=treatment.function(SOURCE,'_graft_lt_run')
        self.assertLess(run_source.index('_pr_graph_before_read'),run_source.index('lt.call('))
        self.assertNotIn('CU_GRAPH_USER_OBJECT_MOVE',treatment.HOST)
        for old,new in treatment.transforms(ORIGINAL):
            with self.subTest(transform=old.splitlines()[0]):
                with self.assertRaises(ValueError):
                    treatment.validate(SOURCE.replace(new,old,1),ORIGINAL)


if __name__=='__main__':
    unittest.main()
