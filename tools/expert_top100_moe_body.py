"""Standalone adapter body; embedded after the exact pinned native dependency closure."""


def _moe_global_cache(tokens):
    raise RuntimeError('EXPERT_MOE_STATE_REQUIRED')


def _expert_moe_budget(deadline):
    if time.monotonic() >= deadline:
        raise RuntimeError('EXPERT_MOE_PREPARE_295S')


def _expert_moe_tensor(tensor, shape, dtype, device, name):
    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype != dtype or tensor.device != device or not tensor.is_cuda
            or not tensor.is_contiguous() or tensor.data_ptr() % 16):
        raise RuntimeError('EXPERT_MOE_TENSOR_CONTRACT: ' + name)


def _expert_moe_weights(weights):
    specs = {
        'w13': ((256, 8192, 2048), torch.uint8),
        'w13_sf': ((256, 8192, 256), torch.float8_e4m3fn),
        'w13_gs': ((256,), torch.float32),
        'w2': ((256, 4096, 2048), torch.uint8),
        'w2_sf': ((256, 4096, 256), torch.float8_e4m3fn),
        'w2_gs': ((256,), torch.float32),
        'post_norm_w': ((4096,), torch.bfloat16),
    }
    if not isinstance(weights, dict) or not specs.keys() <= weights.keys():
        raise RuntimeError('EXPERT_MOE_WEIGHT_KEYS')
    if not isinstance(weights['w13'], torch.Tensor):
        raise RuntimeError('EXPERT_MOE_TENSOR_CONTRACT: w13')
    device = weights['w13'].device
    for name, (shape, dtype) in specs.items():
        _expert_moe_tensor(weights[name], shape, dtype, device, name)
    local = {name: weights[name][:32].clone() for name in specs if name != 'post_norm_w'}
    local['post_expert_norm'] = weights['post_norm_w'].clone()
    return local


def _expert_moe_guarded(tensor, state):
    nbytes = tensor.numel() * tensor.element_size()
    words = (nbytes + 3) // 4
    full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL,
                      dtype=torch.int32, device=tensor.device)
    full.view(torch.uint8)[:nbytes].copy_(tensor.reshape(-1).view(torch.uint8))
    view = full.view(torch.uint8)[:nbytes].view(tensor.dtype).view(tensor.shape)
    state['guards'].append((full, words))
    return view


def _expert_moe_check_guards(state):
    torch.cuda.synchronize(state['device'])
    for full, words in state['guards']:
        if bool((full[words:] != SCRATCH_SENTINEL).any()):
            raise RuntimeError('EXPERT_MOE_SCRATCH_OVERRUN')


def _expert_moe_prepare(weights, arm, deadline=None):
    if type(arm) is not bool:
        raise RuntimeError('EXPERT_MOE_ARM_TYPE')
    if deadline is None:
        deadline = time.monotonic() + 295
    _expert_moe_budget(deadline)
    local = _expert_moe_weights(weights)
    device = local['w13'].device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('EXPERT_MOE_PREPARE_DURING_CAPTURE')
        properties = torch.cuda.get_device_properties(device)
        if ((properties.major, properties.minor) != (9, 0)
                or not 0 < properties.multi_processor_count <= 136):
            raise RuntimeError('EXPERT_MOE_NATIVE_DEVICE_CONTRACT')
        state = dict(device=device, nsm=properties.multi_processor_count,
                     ma_ext={}, guards=[], identities=EXPERT_MOE_IDENTITIES)
        state['ma_sfx'] = _MA_SFX_IF_PRMT if (
            _MA_PRMT_SCALES and _ma_scales_prmt_exact(local['w13_sf'], local['w2_sf'])) else ()
        if not state['ma_sfx']:
            raise RuntimeError('EXPERT_MOE_NATIVE_SCALE_CONTRACT_NO_FALLBACK')
        state['ma_sc6'] = bool(SC6_WGS and state['ma_sfx'] and _wgs_on()
                               and _sc6_ok(local['w13_sf'], local['w2_sf']))
        _expert_moe_budget(deadline)
        _expert_pair_bind(state, arm)
        _expert_moe_budget(deadline)
        for tokens in (1, 4, 8, 16, 32, 33, 128, 256, 257):
            _fork_native_moe(tokens, state)
            _expert_moe_budget(deadline)
        _fork_native_ma_prepare(local, state)
        torch.cuda.synchronize(device)
        _expert_moe_budget(deadline)
        if set(state['ma_ext']) != {'tiny', 'tiny2', 'mid', 'b4', 'main'}:
            raise RuntimeError('EXPERT_MOE_INCOMPLETE_BUILD')
        _expert_moe_check_guards(state)
        _expert_moe_budget(deadline)
        return state


def prepare(weights):
    return _expert_moe_prepare(weights, False)


def expert_moe_prepare_candidate(weights):
    return _expert_moe_prepare(weights, True)


def expert_moe_prepare_four(weights):
    deadline = time.monotonic() + 295
    return tuple(_expert_moe_prepare(weights, arm, deadline)
                 for arm in (False, False, True, True))


def _expert_moe_run(data, state, arm):
    if (state.get('identities') != EXPERT_MOE_IDENTITIES
            or state.get('_expert_pair_arm') is not arm
            or state.get('_expert_pair_arm_bound') is not arm):
        raise RuntimeError('EXPERT_MOE_STATE_IDENTITY_OR_ARM')
    if not isinstance(data, dict) or not {'config', 'hidden_states', 'topk_ids', 'topk_weights'} <= data.keys():
        raise RuntimeError('EXPERT_MOE_DATA_KEYS')
    config = data['config']
    expected = dict(d_hidden=4096, d_expert=4096, n_experts=256,
                    local_start=0, local_count=32, top_k=8)
    if not isinstance(config, dict) or any(type(config.get(key)) is not int or config[key] != value
                                         for key, value in expected.items()):
        raise RuntimeError('EXPERT_MOE_CONFIG_UNSUPPORTED_LOCAL_OR_DIMENSION')
    tokens = config.get('bs')
    if type(tokens) is not int or tokens not in (1, 4, 8, 16, 32, 33, 128, 256, 257):
        raise RuntimeError('EXPERT_MOE_TOKEN_CONTRACT')
    device = state['device']
    inputs = (('hidden_states', (tokens, 4096), torch.bfloat16),
              ('topk_ids', (tokens, 8), torch.int32),
              ('topk_weights', (tokens, 8), torch.float32))
    for name, shape, dtype in inputs:
        _expert_moe_tensor(data[name], shape, dtype, device, name)
    if torch.cuda.current_device() != device.index:
        raise RuntimeError('EXPERT_MOE_CURRENT_DEVICE')
    output = torch.empty_like(data['hidden_states'])
    _fork_native_moe(tokens, state).okc_run(
        state['ma_w'], state['ma_scratch'], data['hidden_states'], data['topk_ids'],
        data['topk_weights'], output, 0, 32, state['nsm'])
    return output


def kernel_fn(data, state):
    return _expert_moe_run(data, state, False)


def expert_moe_candidate_fn(data, state):
    return _expert_moe_run(data, state, True)
