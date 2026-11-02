"""Host-only T1 binding, appended to the unchanged original-control closure."""


def _t1_install(state):
    if tuple(state['ma_sfx']) != ('-DOKC_PRMTSC=1',) or len(state['ma_w']) != 14:
        raise RuntimeError('T1_MOE_PACKING_STATE')
    state['t1_guards'] = []
    state['t1_scratch'] = []
    if state['_t1_arm']:
        state['t1_module'] = _RouterExt(load_inline(
            name='t1_three_launch_v1', cpp_sources=_T1_THREE_CPP,
            cuda_sources=_T1_THREE_CUDA, functions=['t1_three_run'],
            extra_cflags=['-O3', '-std=c++20'],
            extra_cuda_cflags=_T1_THREE_FLAGS, verbose=False))
        temporary = dict(guards=state['t1_guards'])
        for shape, dtype in (((8, 4096), torch.bfloat16),
                             ((8, 4096), torch.float32),
                             ((8, 256), torch.float32)):
            tensor = torch.zeros(shape, dtype=dtype, device=state['device'])
            state['t1_scratch'].append(_expert_moe_guarded(tensor, temporary))


def _t1_selected(tokens, state):
    if (state.get('_t1_arm') is not state.get('_t1_arm_bound')
            or type(state.get('_t1_arm')) is not bool
            or '_expert_pair_arm' in state or '_expert_pair_arm_bound' in state):
        raise RuntimeError('T1_MOE_STALE_OR_CROSSED_ARM')
    if tokens == 1 and state['_t1_arm']:
        if len(state['t1_scratch']) != 3 or len(state['t1_guards']) != 3:
            raise RuntimeError('T1_MOE_SCRATCH_INVENTORY')
        return state['t1_module']
    return _fork_native_moe(tokens, state)


def _t1_run(data, state, arm):
    if (state.get('identities') != EXPERT_MOE_IDENTITIES
            or state.get('_t1_arm') is not arm
            or state.get('_t1_arm_bound') is not arm):
        raise RuntimeError('EXPERT_MOE_STATE_IDENTITY_OR_ARM')
    _expert_moe_data(data, state)
    tokens = data['config']['bs']
    output = torch.empty_like(data['hidden_states'])
    module = _t1_selected(tokens, state)
    if tokens == 1 and arm:
        module.t1_three_run(state['ma_w'], state['t1_scratch'],
            data['hidden_states'], data['topk_ids'], data['topk_weights'], output)
    else:
        module.okc_run(state['ma_w'], state['ma_scratch'],
            data['hidden_states'], data['topk_ids'], data['topk_weights'],
            output, 0, 32, state['nsm'])
    return output
