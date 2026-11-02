"""Actual-entry inherited gates and helper-scope failure/restore source contracts."""

import re
import unittest

import g_pr_graph_owner_relaxed as treatment
import test_g_pr_graph_owner_events as events


events.legacy.treatment = treatment
events.legacy.SOURCE = treatment.build(events.legacy.ORIGINAL)
events.treatment = treatment


class ScopeContracts(events.EventContracts):
    def test_scope_exact_previous_modes_and_error_exits(self):
        for previous in ('GLOBAL','THREAD_LOCAL','RELAXED'):
            for fail in (None,'enter','context-push','owner-completion','unmap','context-pop','primary-release','restore'):
                with self.subTest(previous=previous,fail=fail):
                    sequence=[]
                    mode=previous
                    caller_mode='GLOBAL'
                    if fail=='enter':
                        sequence.append('entry-fatal')
                    else:
                        mode='RELAXED'
                        sequence.append('enter')
                        for stage in ('context-push','owner-completion','unmap','context-pop','primary-release'):
                            sequence.append(stage)
                            if stage==fail:
                                mode=previous
                                sequence.extend(('restore-exact','fatal-no-retry'))
                                break
                        else:
                            if fail=='restore':
                                sequence.append('restore-fatal-no-continue')
                            else:
                                mode=previous
                                sequence.append('restore-exact')
                    self.assertEqual(caller_mode,'GLOBAL')
                    if fail!='restore':
                        self.assertEqual(mode,previous)
                    if fail=='enter':
                        self.assertNotIn('unmap',sequence)
                    self.assertEqual(sequence.count('primary-release'),int(fail not in ('enter','context-push','owner-completion','unmap','context-pop')))

    def test_actual_helper_scope_and_failure_restore_source(self):
        host=treatment.HOST
        helper=re.search(r'static void retirement.*?\n}',host,re.S).group()
        self.assertLess(helper.index('callback_finished.load'),helper.index('cleanup-mode-enter'))
        self.assertLess(helper.index('cleanup-mode-enter'),helper.index('mapping->cleanup(&previous)'))
        self.assertLess(helper.index('mapping->cleanup(&previous)'),helper.index('cleanup-mode-restore'))
        self.assertLess(helper.index('cleanup-mode-restore'),helper.index('delete mapping'))
        cleanup=host[host.index('  void cleanup('):host.index('\n};')]
        self.assertIn('CUstreamCaptureMode restore=*previous;',cleanup)
        self.assertLess(cleanup.index('cleanup-mode-error-restore'),cleanup.index('fatal(result, stage)'))
        self.assertIn('popped != context',cleanup)
        self.assertEqual(host.count('mapping->cleanup(&previous)'),1)
        self.assertIn('owner->cleanup();',host)
        self.assertNotIn('CU_STREAM_CAPTURE_MODE_GLOBAL',host)
        for name in ('_graft_lt_run','_graft_lt_plan'):
            self.assertNotIn('ExchangeStreamCaptureMode',treatment.function(events.legacy.SOURCE,name))
        callback=re.search(r'static void CUDA_CB graph_release.*?\n}',host,re.S).group()
        self.assertNotIn('cuThread',callback)
        self.assertNotIn('cuCtxSynchronize',host)
        self.assertNotIn('cuStreamSynchronize',host)


if __name__ == '__main__':
    unittest.main()

