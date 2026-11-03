"""Actual generated T1 prepare/dispatch and CPU-only range/math contracts."""

import ast
import contextlib
import copy
import io
import math
import random
import struct
import textwrap
import types
import unittest

import g_t1_three_launch as treatment
from prod_size_compile_g import bindings
from test_g_pr_compressed_owner import Tensor, actual_entries
from test_g_timed_silu_table import capture

ORIGINAL = treatment.BASE.read_bytes().decode()
SOURCE = treatment.build(ORIGINAL)
NATIVE = treatment.native_source(ORIGINAL)


def function(source, name):
    node = [node for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name][-1]
    return ast.get_source_segment(source, node)


class Buffer(Tensor):
    def __init__(self, shape, dtype='bf16', device='cpu-mock-cuda-device', contiguous=True):
        super().__init__(shape, dtype)
        self.device, self.contiguous_flag = device, contiguous

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.contiguous_flag

    def __deepcopy__(self, memo):
        result = type(self)(self.shape, self.dtype, self.device, self.contiguous_flag)
        memo[id(self)] = result
        return result

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError('row slices only')
        rows = len(range(self.shape[0])[key])
        return type(self)((rows,) + self.shape[1:], self.dtype, self.device, self.contiguous_flag)


def integrated(suffix=None, eligible=True, scale_suffix=('-DOKC_PRMTSC=1',)):
    env, events, captured = actual_entries()
    native_lt = next(node for node in ast.parse(ORIGINAL).body
                     if isinstance(node,ast.ClassDef) and node.name=='_GraftLt')
    deepcopy_method = next(node for node in native_lt.body
                          if isinstance(node,ast.FunctionDef) and node.name=='__deepcopy__')
    method_env = {}
    exec(textwrap.dedent(ast.get_source_segment(ORIGINAL,deepcopy_method)),method_env)
    env['_GraftLt'].__deepcopy__ = method_env['__deepcopy__']
    allocations, builds, calls = [], [], []
    previous = env['_vj_prepare']

    def normal_prepare(weights):
        state = previous(weights)
        state.update(_fork_native_eligible=eligible, ma_sfx=scale_suffix,
                     ma_w=[Buffer((1,)) for index in range(12)], ma_scratch=[object()], ma_sc6=True,nsm=132)
        return state

    class OriginalModule:
        def okc_run(self, *args):
            calls.append(('original', args))

    class Module:
        def t1_three_run(self, *args):
            calls.append(('three', args))

    original = OriginalModule()
    env['_vj_prepare'] = normal_prepare
    env['_fork_native_moe'] = lambda tokens, state: original
    env['_fork_moe_fallback'] = lambda tokens, state: original
    exec(function(ORIGINAL, '_moe'), env)
    exec(function(ORIGINAL, '_RouterExt'), env)

    def empty(shape, **kwargs):
        value = Buffer(shape, kwargs['dtype'], kwargs['device'])
        allocations.append(value)
        return value

    def load_inline(**kwargs):
        builds.append(kwargs)
        self_test = list(events)
        if self_test != ['normal_prepare', 'complete_prior_work', 'original_startup'] * len(builds):
            raise AssertionError('activation/build before real final prepare returned')
        return Module()

    env['torch'].empty = empty
    env['torch'].int32 = 'i32'
    env['load_inline'] = load_inline
    env['_t1_three_test_original'] = original
    exec(suffix if suffix is not None else SOURCE[len(ORIGINAL):], env)
    return env, events, allocations, builds, calls


def prepare(env):
    return env['prepare']({'w13':Buffer((32,8192,2048))})


def run(env, state, tokens=1, local_start=0, local_count=32, input_dtype='bf16', contiguous=True):
    module = env['_moe'](tokens,state)
    input_tensor = Buffer((tokens,4096),input_dtype,contiguous=contiguous)
    ids = Buffer((tokens,8),'i32')
    weights = Buffer((tokens,8),'f32')
    output = Buffer((tokens,4096))
    module.okc_run(state['ma_w'],state['ma_scratch'],input_tensor,ids,weights,output,
                   local_start,local_count,132)
    return input_tensor, ids, weights, output


def f32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def bf16(value):
    bits = struct.unpack('<I',struct.pack('<f',f32(value)))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f',struct.pack('<I',bits))[0]


def gate_model(gate, up, scale, round_inputs=True):
    gate = f32(gate * scale)
    up = f32(up * scale)
    if round_inputs:
        gate, up = bf16(gate), bf16(up)
    return bf16(f32(f32(gate / f32(1.0 + f32(math.exp(-gate)))) * up))


def down_model(value, scale, norm, rounded_down=True, rounded_gamma=True):
    down = f32(value * scale)
    gamma = f32(1.0 + bf16(norm))
    return f32((bf16(down) if rounded_down else down) * (bf16(gamma) if rounded_gamma else gamma))


def tile_partial(values):
    lanes = [f32(f32(values[group]*values[group]) + f32(values[group+8]*values[group+8]))
             for group in range(8)]
    for offset in (1,2,4):
        lanes = [f32(value + lanes[group ^ offset]) for group,value in enumerate(lanes)]
    return lanes[0]


def inverse_model(values):
    partials = [tile_partial(values[start:start+16]) for start in range(0,4096,16)]
    warp_sums = []
    for start in range(0,256,32):
        lanes = partials[start:start+32]
        for offset in (16,8,4,2,1):
            lanes = [f32(value + lanes[index+offset]) if index+offset<32 else value
                     for index,value in enumerate(lanes)]
        warp_sums.append(lanes[0])
    total = 0.0
    for value in warp_sums:
        total = f32(total + value)
    return f32(1.0 / math.sqrt(f32(f32(total * (1.0/4096)) + 1e-8)))


def combine_model(ids, route_weights, expert_y, slot_order=range(8)):
    inverse = {slot:inverse_model(expert_y[slot]) for slot,expert in enumerate(ids) if 0<=expert<32}
    output = []
    for column in range(4096):
        combined = 0.0
        for slot in slot_order:
            if slot in inverse:
                weight = f32(route_weights[slot]*inverse[slot])
                combined = f32(combined + f32(weight*expert_y[slot][column]))
        output.append(bf16(combined))
    return output


class Contracts(unittest.TestCase):
    def test_actual_layer_expert_dispatch_block_t1_and_t8(self):
        env,events,allocations,builds,calls=integrated()
        state=prepare(env)
        original=ast.parse(function(ORIGINAL,'_gle_after_attention')).body[0]
        start=next(index for index,node in enumerate(original.body)
                   if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                   and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='empty'
                   and any(isinstance(target,ast.Name) and target.id=='m' for target in node.targets))
        block=ast.Module(body=original.body[start:start+4],type_ignores=[])
        self.assertIsInstance(block.body[3],ast.For)
        has_pdl=bindings(ORIGINAL)['MOE_HAS_PDL']
        self.assertIs(has_pdl,False)
        for tokens in (1,8):
            env.update(state=state,T=tokens,dev='cpu-mock-cuda-device',D_EXPERT=4096,
                xm=Buffer((tokens,4096)),lids=Buffer((tokens,8),'i32'),wts=Buffer((tokens,8),'f32'),
                MOE_HAS_PDL=has_pdl,_MA_MMAX=384)
            exec(compile(block,'<actual selected layer expert block>','exec'),env)
            self.assertEqual(calls[-1][0],'three' if tokens==1 else 'original')

    def test_exact_prefix_helpers_and_all_seven_factories(self):
        manifest=treatment.validate(SOURCE,ORIGINAL)
        self.assertTrue(manifest['exact_reverse'])
        self.assertEqual(SOURCE[:len(ORIGINAL)],ORIGINAL)
        before,old_routes=capture(ORIGINAL)
        after,new_routes=capture(SOURCE)
        self.assertEqual(len(before),7)
        self.assertEqual(before,after)
        self.assertEqual(old_routes,new_routes)
        self.assertEqual(before[0]['extra_cuda_cflags'],treatment.FLAGS)
        helper=bindings(ORIGINAL)['_MA_ALT_SRCS']['moe_helper0']
        for name in treatment.HELPERS:
            self.assertEqual(treatment.cuda_function(NATIVE,name),treatment.cuda_function(helper,name))

    def test_actual_final_prepare_dispatch_and_independent_states(self):
        env,events,allocations,builds,calls=integrated()
        first,second=prepare(env),prepare(env)
        self.assertEqual(events,['normal_prepare','complete_prior_work','original_startup']*2)
        self.assertEqual(len(builds),2)
        self.assertEqual(builds[0]['cuda_sources'],NATIVE)
        self.assertEqual(builds[0]['extra_cuda_cflags'],treatment.FLAGS)
        self.assertEqual(builds[0]['extra_cflags'],['-O3','-std=c++20'])
        self.assertEqual([tensor.shape for tensor in allocations],[(8,4096),(8,4096),(8,256)]*2)
        self.assertEqual([tensor.dtype for tensor in allocations],['bf16','f32','f32']*2)
        self.assertEqual(sum(tensor.numel()*(2 if tensor.dtype=='bf16' else 4) for tensor in allocations[:3]),204800)
        for state in (first,second,first):
            tensors=run(env,state)
            self.assertEqual(calls[-1][0],'three')
            self.assertIs(calls[-1][1][0],state['ma_w'])
            self.assertIs(calls[-1][1][1],state['_t1_three_adapter'].scratch)
            self.assertEqual(calls[-1][1][2:],tensors)
        self.assertIsNot(first['_t1_three_adapter'].scratch[0],second['_t1_three_adapter'].scratch[0])
        self.assertEqual(len(allocations),6)

    def test_actual_t8_all_other_sizes_and_unsupported_fallback(self):
        env,events,allocations,builds,calls=integrated()
        state=prepare(env)
        for tokens in (8,16,32,33,128,256,384):
            run(env,state,tokens)
            self.assertEqual(calls[-1][0],'original')
        for options in (dict(local_start=1),dict(local_count=16),dict(input_dtype='f32'),dict(contiguous=False)):
            run(env,state,**options)
            self.assertEqual(calls[-1][0],'original')
        for eligible,suffix in ((False,('-DOKC_PRMTSC=1',)),(True,()),(True,('-DOKC_PRMTSC=0',))):
            env,events,allocations,builds,calls=integrated(eligible=eligible,scale_suffix=suffix)
            state=prepare(env)
            self.assertNotIn('_t1_three_adapter',state)
            self.assertEqual(builds,[])
            self.assertEqual(allocations,[])
            run(env,state)
            self.assertEqual(calls[-1][0],'original')

    def test_deepcopy_and_prepare_failure_no_activation(self):
        env,events,allocations,builds,calls=integrated()
        state=prepare(env)
        cloned=copy.deepcopy(state)
        self.assertIsNot(state['_t1_three_adapter'].scratch[0],cloned['_t1_three_adapter'].scratch[0])
        self.assertIs(state['_t1_three_adapter'].module,cloned['_t1_three_adapter'].module)
        run(env,cloned)
        self.assertIs(calls[-1][1][1],cloned['_t1_three_adapter'].scratch)
        env['_t1_three_previous_prepare']=lambda weights: (_ for _ in ()).throw(RuntimeError('original-prepare-failed'))
        with self.assertRaisesRegex(RuntimeError,'original-prepare-failed'):
            prepare(env)
        self.assertEqual(len(builds),1)

    def test_ranges_slot_mapping_and_padding(self):
        for row_blocks,columns in ((512,8),(256,16)):
            writes=[block*columns+group for block in range(row_blocks) for group in range(columns)]
            self.assertEqual(writes,list(range(4096)))
            last=(31*row_blocks+row_blocks-1)*36864+36863
            self.assertEqual(last+1,32*row_blocks*36864)
        code_bytes={((warp*8+chunk)*512+lane*16+byte)
            for warp in range(8) for chunk in range(8) for lane in range(32) for byte in range(16)}
        self.assertEqual(code_bytes,set(range(32768)))
        scale_bytes={warp*512+lane*16+(chunk//2)*4+(chunk%2)*2+half
            for warp in range(8) for lane in range(32) for chunk in range(8) for half in range(2)}
        self.assertEqual(scale_bytes,set(range(4096)))
        inputs={warp*512+16*token_pair+chunk*64+half*8+element
            for warp in range(8) for token_pair in range(4) for chunk in range(8)
            for half in range(2) for element in range(8)}
        self.assertEqual(inputs,set(range(4096)))
        self.assertIn('group == 0',NATIVE)
        self.assertIn(': make_uint4(0, 0, 0, 0)',NATIVE)
        routes=[-1,0,31,32,255,-100,7,7]
        self.assertEqual([slot for slot,expert in enumerate(routes) if 0<=expert<32],[1,2,6,7])
        self.assertEqual([expert for expert in routes if 0<=expert<32],[0,31,7,7])
        self.assertNotIn('% 32',NATIVE)

    def test_actual_pack_bit_layout_and_selected_scale_embedding(self):
        env={'DQ7':True}
        exec(function(ORIGINAL,'_stored_pos'),env)
        positions=env['_stored_pos']()
        for register in range(4):
            for half in range(2):
                for code in range(16):
                    word=sum(((code>>bit)&1)<<positions[(register,half,bit)] for bit in range(4))
                    rot10=((word<<10)|(word>>22))&0xffffffff
                    rot14=((word<<14)|(word>>18))&0xffffffff
                    decoded=[word&0x81c081c0,(word<<4)&0x81c081c0,rot10&0x81c081c0,
                             ((word<<6)&0x80c080c0)|(rot14&0x01000100)]
                    expected=((code&1)<<6)|((code&2)<<6)|((code&4)<<6)|((code&8)<<12)
                    self.assertEqual((decoded[register]>>(16*half))&65535,expected)
        for scale in range(8,127):
            value=(1+(scale&7)/8.0)*2.0**((scale>>3)-7)
            actual=struct.unpack('<I',struct.pack('<f',bf16(value*2.0**64)))[0]>>16
            self.assertEqual(actual,scale*16+0x5c00)

    def test_bf16_boundaries_and_no_global_norm(self):
        self.assertEqual(bf16(1.00390625),1.0)
        self.assertEqual(bf16(1.01171875),1.015625)
        self.assertEqual(struct.pack('<f',bf16(-0.0)),struct.pack('<f',-0.0))
        generator=random.Random(701)
        self.assertTrue(any(gate_model(value,1.003,1.0)!=gate_model(value,1.003,1.0,False)
                            for value in [generator.uniform(-3,3) for index in range(100)]))
        self.assertNotEqual(down_model(1.003,1.0,0.00390625),down_model(1.003,1.0,0.00390625,False))
        self.assertNotEqual(down_model(1.0,1.0,0.00390625),down_model(1.0,1.0,0.00390625,True,False))
        first=[1.0]*4096
        second=[100.0 if column<2048 else -50.0 for column in range(4096)]
        output=combine_model([4,9,-1,-1,-1,-1,-1,-1],[0.3,0.7,0,0,0,0,0,0],{0:first,1:second})
        mixed=[f32(.3*left+.7*right) for left,right in zip(first,second)]
        wrong_inverse=inverse_model(mixed)
        wrong=[bf16(f32(value*wrong_inverse)) for value in mixed]
        self.assertNotEqual(output,wrong)

    def test_empty_varied_routes_stale_scratch_and_slot_order(self):
        class Poison:
            def __getitem__(self,key):
                raise AssertionError('invalid route read stale/unwritten scratch')
        empty=combine_model([-1,32,33,-9,255,64,-2,999],[1.0]*8,Poison())
        self.assertEqual(empty,[0.0]*4096)
        positive=[1.0]*4096
        tiny=[1e-8]+[1.0]*4095
        negative=[-1.0]*4096
        ids=[31,0,7,-1,-1,-1,-1,-1]
        weights=[1.0/3]*3+[0.0]*5
        values={0:positive,1:tiny,2:negative}
        ordered=combine_model(ids,weights,values)
        reordered=combine_model(ids,weights,values,slot_order=(0,2,1,3,4,5,6,7))
        self.assertNotEqual(ordered[0],reordered[0])
        for routes in ([0,-1,-1,-1,-1,-1,-1,-1],[-1]*8,[7,31,-1,0,-1,-1,-1,-1]):
            selected={slot:[float(slot+1)]*4096 for slot,expert in enumerate(routes) if 0<=expert<32}
            result=combine_model(routes,[.125]*8,selected)
            self.assertEqual(len(result),4096)
            self.assertTrue(all(math.isfinite(value) for value in result))

    def test_all_local_pair_counts_zero_through_eight(self):
        stale={slot:[1.0]*4096 for slot in range(8)}
        for count in (8,0,1,7,2,6,3,5,4,0,8):
            routes=list(range(count))+[-1]*(8-count)
            output=combine_model(routes,[.125]*8,stale)
            self.assertEqual(output,[bf16(count*.125)]*4096)

    def test_mutation_negatives_and_three_ordered_launches(self):
        mutations=[('M == 1','M <= 8'),('local_route(params, slot)','true'),
                   ('params.ids[slot] - params.local_start','params.ids[slot] % 32'),
                   ('fork_bf16(gate_or_first * global_scale)','(gate_or_first * global_scale)'),
                   ('fork_bf16(1.f + __bfloat162float(params.norm[column]))','(1.f + __bfloat162float(params.norm[column]))'),
                   ('rsqrtf(square_sum * (1.f / 4096) + 1e-8f)','1.f')]
        for before,after in mutations:
            changed=SOURCE.replace(before,after,1)
            self.assertNotEqual(changed,SOURCE,before)
            with self.assertRaises(ValueError):
                treatment.validate(changed,ORIGINAL)
            if before in NATIVE:
                with self.assertRaises(ValueError):
                    treatment.native_contract(NATIVE.replace(before,after,1))
        bad=treatment.appendix(ORIGINAL).replace('M == 1','M <= 8',1)
        env,events,allocations,builds,calls=integrated(suffix=bad)
        state=prepare(env)
        self.assertIs(env['_moe'](8,state),state['_t1_three_adapter'])
        env,events,allocations,builds,calls=integrated()
        state=prepare(env)
        self.assertIsNot(env['_moe'](8,state),state['_t1_three_adapter'])
        self.assertEqual(NATIVE.count('<<<'),3)
        self.assertLess(NATIVE.index('gate_up<<<'),NATIVE.index('down<<<'))
        self.assertLess(NATIVE.index('down<<<'),NATIVE.index('norm_combine<<<'))
        self.assertEqual(NATIVE.count('cudaStream_t stream = at::cuda::getCurrentCUDAStream();'),1)
        self.assertNotIn('atomicAdd',NATIVE)
        self.assertNotIn('while (',NATIVE)
        self.assertNotIn('cudaStreamIsCapturing',NATIVE)


if __name__ == '__main__':
    unittest.main()
