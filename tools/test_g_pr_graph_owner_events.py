"""Actual-entry CPU contracts for the narrow owner-event amendment."""

import gc
import re
import unittest

import g_pr_graph_owner_events as treatment
import test_g_pr_graph_owner as legacy

legacy.treatment = treatment
legacy.SOURCE = treatment.build(legacy.ORIGINAL)
original_before = legacy.Owner.before_read
original_drain = legacy.Module.drain


def before_read(self, stream):
    original_before(self, stream)
    if self.module.graph is not None:
        return 0
    event = dict(owner=self.record, stream=stream, recorded=False)
    if not hasattr(self.record, 'completions'):
        self.record.completions = []
    self.record.completions.append(event)
    return event


def after_read(self, stream, event):
    if event:
        if event['owner'] is not self.record or event['stream'] != stream:
            raise RuntimeError('wrong completion owner/stream')
        if getattr(self.module, 'record_failure', False):
            self.module.fatal = True
            raise RuntimeError('fatal-event-record-no-cleanup')
        event['recorded'] = True


def drain(self, fail_completion=False):
    if getattr(self, 'fatal', False):
        self.worker_events.append('fatal-abort-no-unmap')
        return
    for record in self.retirement:
        for event in getattr(record, 'completions', []):
            if not event['recorded']:
                raise RuntimeError('unrecorded-owner-event')
    return original_drain(self, fail_completion)


legacy.Owner.before_read = before_read
legacy.Owner.after_read = after_read
legacy.Module.drain = drain


class EventContracts(legacy.Contracts):
    def test_delayed_a_retirement_does_not_encompass_b_capture(self):
        module = legacy.Module()
        env, unused_events, unused_captured = legacy.integrated(module)
        first, second = env['prepare']({}), env['prepare']({})
        legacy.run(env, first, 128)
        record = first['graft_lt'][('pr',128)]['_pr_graph_owner'].record
        self.assertTrue(record.completions[0]['recorded'])
        del first
        gc.collect()
        graph = legacy.Graph()
        module.graph = graph
        legacy.run(env, second, 256)
        module.drain()
        self.assertEqual(module.graph, graph)
        self.assertEqual(len(graph.records), 1)
        self.assertNotIn('cuCtxSynchronize', treatment.HOST)
        self.assertNotIn('cuStreamSynchronize', treatment.HOST)
        self.assertIn('cuEventSynchronize(event)', treatment.HOST)
        graph.destroy()

    def test_after_hook_covers_lt_exception_and_both_plans(self):
        module = legacy.Module()
        env, unused_events, captured = legacy.integrated(module)
        state = env['prepare']({})
        for tokens in (128,256):
            plan = state['graft_lt'][('pr',tokens)]
            call = plan['lt'].call
            def failure(*args):
                call(*args)
                raise RuntimeError('lt-failed-after-enqueue')
            plan['lt'].call = failure
            try:
                with self.assertRaisesRegex(RuntimeError, 'lt-failed-after-enqueue'):
                    legacy.run(env,state,tokens)
                self.assertTrue(plan['_pr_graph_owner'].record.completions[-1]['recorded'])
            finally:
                plan['lt'].call = call
        self.assertEqual(len(captured),2)

    def test_fresh_events_and_capture_exclusion(self):
        module = legacy.Module()
        env, unused_events, unused_captured = legacy.integrated(module)
        state = env['prepare']({})
        owner = state['graft_lt'][('pr',128)]['_pr_graph_owner']
        legacy.run(env,state,128)
        legacy.run(env,state,256)
        self.assertEqual(len(owner.record.completions),2)
        self.assertIsNot(owner.record.completions[0], owner.record.completions[1])
        module.graph = legacy.Graph()
        legacy.run(env,state,128)
        self.assertEqual(len(owner.record.completions),2)
        module.graph.destroy()

    def test_record_failure_is_fatal_not_unmap_or_fallback(self):
        module = legacy.Module()
        env, unused_events, unused_captured = legacy.integrated(module)
        state = env['prepare']({})
        module.record_failure = True
        with self.assertRaisesRegex(RuntimeError, 'fatal-event-record'):
            legacy.run(env,state,128)
        del state
        gc.collect()
        module.drain()
        self.assertNotIn('unmap',module.worker_events)

    def test_independent_stream_event_ownership(self):
        module = legacy.Module()
        env, unused_events, unused_captured = legacy.integrated(module)
        first, second = env['prepare']({}), env['prepare']({})
        for state, stream in ((first,19),(second,23),(first,29)):
            env['torch'].cuda.current_stream = lambda value=stream: type('Stream',(),{'cuda_stream':value})()
            legacy.run(env,state,128)
        owner_a = first['graft_lt'][('pr',128)]['_pr_graph_owner'].record
        owner_b = second['graft_lt'][('pr',128)]['_pr_graph_owner'].record
        self.assertEqual([event['stream'] for event in owner_a.completions],[19,29])
        self.assertEqual([event['stream'] for event in owner_b.completions],[23])

    def test_source_copy_completion_and_finally_failure_policy(self):
        host = treatment.HOST
        self.assertLess(host.index('cuMemcpyDtoDAsync'),host.index('cuEventRecord(copy_event'))
        self.assertLess(host.index('cuEventRecord(copy_event'),host.index('cuEventSynchronize(copy_event'))
        self.assertLess(host.index('cuEventSynchronize(copy_event'),host.index('require(copy_result'))
        self.assertIn('fatal(cuEventRecord',host)
        self.assertIn('fatal(cuEventDestroy(event)',host)
        run = treatment.function(legacy.SOURCE,'_graft_lt_run')
        self.assertIn('finally:\n        _pr_graph_after_read(ticket, stream)',run)
        self.assertLess(run.index('lt.call('),run.index('_pr_graph_after_read'))
        callback = re.search(r'static void CUDA_CB graph_release.*?\n}',host,re.S).group()
        self.assertNotIn('cuEvent',callback)


if __name__ == '__main__':
    unittest.main()

