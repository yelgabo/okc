import contextlib
import io
import json

from test_moe_phase_host import Host
from expert_top100_moe_test import prepare_small

Host.setUpClass()
test = Host()
namespace, runtime = test.environment()
reset = namespace['_h_reset']
budget = namespace['_expert_moe_budget']
cleaned = False


def reset_then_flag(states):
    global cleaned
    reset(states)
    cleaned = True


def budget_after_cleanup(deadline):
    if cleaned:
        raise RuntimeError('SYNTHETIC_POST_CLEANUP_BUDGET')
    budget(deadline)


namespace['_h_reset'] = reset_then_flag
namespace['_expert_moe_budget'] = budget_after_cleanup
output = io.StringIO()
try:
    with contextlib.redirect_stdout(output):
        prepare_small(namespace, '_phase_host_entry')
except RuntimeError as error:
    assert str(error) == 'SYNTHETIC_POST_CLEANUP_BUDGET', str(error)
    result = dict(failure=str(error), cleanup_completed=cleaned,
                  retained_partial='MOE_PHASE_PARTIAL ' in output.getvalue(),
                  scope='Synthetic CPU failure path, no CUDA timing evidence')
    print(json.dumps(result, indent=2))
else:
    raise AssertionError('Injected budget failure was lost')
