"""Frozen-source integer/event contracts, not CUDA schedule or numerical validation."""
import bisect
import collections
import gzip
import json
from pathlib import Path
import re
import unittest

from g_expert_g224_prefix import sha
from test_g_expert_g224_chunk_phased import NATIVE, contract
from test_g_expert_g224_prefix import body, expression

FROZEN_SHA = '53d09576f86e93b9baaa0145dcbdf1b60d8736743677d0a4de60e89864d18492'


def require(source):
    assert sha(source) == FROZEN_SHA
    assert contract(source)
    for text in ('#define OKC_WGSDFIN 2', '#define OKC_MFQ 47', '#define OKC_MFP 15',
                 'if (nj == 0) return 0;', 'if (wt == 0) *fc = 0;',
                 'if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);',
                 'wgs_flush<PB>(p, S, pd);', 'for (int n = 0; ld_acquire(cp) < target && n < 4000000;',
                 'const int target = 64 *', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor',
                 'cudaLaunchCooperativeKernel', 'resident>=2'):
        assert text in source, text
    phase = body(source, '__device__ __forceinline__ void wgs_phase(')
    assert phase.rstrip().endswith('wgs_flush<PB>(p, S, pd);\n}')
    segment = body(source, '__device__ __forceinline__ void wgs_seg(')
    assert segment.index('wg_wait<0>();') < segment.index('mbar_arrive1(S.bars + BX_E + xs)')
    flush = body(source, '__device__ __forceinline__ void wgs_flush(')
    assert flush.index('fence.proxy.async.global;') < flush.index('bar.sync %0, 128;') < flush.index('red.release.gpu.global.add.s32')
    assert source.count('S.wc[') == body(source, '__device__ void route(').count('S.wc[')
    return True


class SourceModel:
    def __init__(self, counts, phase, grid):
        self.counts, self.phase, self.grid = counts, phase, grid
        macro = 'OKC_WGSB' if phase else 'OKC_WGSA'
        self.costs = [int(value) for value in re.search(r'#define '+macro+r' ([0-9, ]+)', NATIVE)[1].split(',')]
        self.unit = int(re.search(r'#define '+('OKC_WGSUB' if phase else 'OKC_WGSU')+r' (\d+)', NATIVE)[1])
        self.units = 32 if phase else 64
        self.prefix = [0]
        for count in counts:
            self.prefix.append(self.prefix[-1] + self.units * (16*self.costs[(count+7)//8-1]+self.unit))
        self.total = self.prefix[-1]
        self.ranges = [(self.stage(self.boundary(cta)), self.units*16*len(counts) if cta==grid-1 else self.stage(self.boundary(cta+1))) for cta in range(grid)]

    def boundary(self, cta):
        function = body(NATIVE, ' uint32_t g224_boundary(')
        return expression(re.search(r'return (.*?);', function)[1], total=self.total, cta=cta, grid=self.grid)

    def stage(self, position):
        if not self.counts:
            return 0
        job = bisect.bisect_right(self.prefix[:-1], position)-1
        cost = self.costs[(self.counts[job]+7)//8-1]
        relative = position-self.prefix[job]
        unit, remainder = divmod(relative, 16*cost+self.unit)
        step = 0 if remainder<self.unit else min(15,(remainder-self.unit)//cost)
        function = body(NATIVE, ' int g224_stage_base(')
        base = expression(re.search(r'return (.*?);',function)[1],job=job,pb=self.phase)
        return base+unit*16+step

    def owner(self, stage):
        job, local = divmod(stage,self.units*16)
        cost = self.costs[(self.counts[job]+7)//8-1]
        end = self.prefix[job]+(local//16)*(16*cost+self.unit)+self.unit+((local&15)+1)*cost
        owner = min(self.grid-1, ((end-1)*self.grid)//self.total)
        while owner+1<self.grid and self.boundary(owner+1)<end:
            owner+=1
        while owner>0 and self.boundary(owner)>=end:
            owner-=1
        return owner

    def segments(self, cta):
        start, end = self.ranges[cta]
        last = ((end-1)>>4)<<4
        split = last if self.phase==0 and end&15 and last>start else start
        result=[]
        for low, high in ((split,end),(start,split)):
            cursor=low
            while cursor<high:
                unit, step=divmod(cursor,16)
                finish=min(16,step+high-cursor)
                result.append((unit,step,finish))
                cursor+=finish-step
        return result


class FrozenContracts(unittest.TestCase):
    def test_identity_source_negatives(self):
        self.assertTrue(require(NATIVE))
        for before,after in (('resident>=2','resident>=1'),('return ub >> 2;','return ub >> 1;'),
                ('if (wt == 0) *fc = 0;','if (wt == 0) *fc = 1;'),
                ('wgs_flush<PB>(p, S, pd);\n}', '\n}'), ('#define OKC_MFQ 47','#define OKC_MFQ 0')):
            with self.assertRaises(AssertionError): require(NATIVE.replace(before,after,1))

    def test_ranges_inverse_split_finisher_and_slots(self):
        cases=[[],[1],[8],[9],[16],[17],[23],[24],[24,1,9],[24]*128,[1]*32+[24]*112]
        cases += [[count] for count in range(1,25)]
        examined=0
        for counts in cases:
            for phase in (0,1):
                for grid in (2,4,16,64,132,264):
                    model=SourceModel(counts,phase,grid)
                    self.assertEqual(model.ranges[0][0],0)
                    self.assertEqual(model.ranges[-1][1],len(counts)*model.units*16)
                    self.assertLess(model.total,2**32)
                    shares=collections.defaultdict(list)
                    for cta,(low,high) in enumerate(model.ranges):
                        self.assertLessEqual(low,high)
                        if cta: self.assertEqual(model.ranges[cta-1][1],low)
                        segments=model.segments(cta)
                        self.assertEqual(sum(end-start for _,start,end in segments),high-low)
                        self.assertEqual(sorted((unit*16+start,unit*16+end) for unit,start,end in segments), sorted((unit*16+max(0,low-unit*16),unit*16+min(16,high-unit*16)) for unit in range(low//16,(high+15)//16)))
                        for unit,start,end in segments:
                            shares[unit].append((cta,start,end))
                            self.assertEqual(model.owner(unit*16+start),cta)
                    slots={}
                    for unit,parts in shares.items():
                        ordered=sorted(parts)
                        self.assertEqual(ordered[0][1],0)
                        self.assertEqual(ordered[-1][2],16)
                        self.assertEqual(sum(end-start for _,start,end in parts),16)
                        self.assertTrue(all(left[2]==right[1] for left,right in zip(ordered,ordered[1:])))
                        if len(parts)>1:
                            slot=model.owner(unit*16)
                            self.assertEqual(slot,ordered[0][0])
                            self.assertLess(slot,265)
                            self.assertNotIn(slot,slots)
                            slots[slot]=unit
                            self.assertEqual(sum(start==0 for _,start,_ in parts),1)
                            if phase:
                                finisher=ordered[0][0]
                                self.assertTrue(all(cta>finisher for cta,_,_ in ordered[1:]))
                        examined+=len(parts)
        self.assertGreater(examined,100000)

    def test_all_even_grids_empty_and_partial_ranges(self):
        for grid in range(2,265,2):
            for phase in (0,1):
                for counts in ([],[1],[24],[1,9,17,24]):
                    model=SourceModel(counts,phase,grid)
                    self.assertEqual(sum(end-start for start,end in model.ranges),len(counts)*model.units*16)
                    for cta,(start,end) in enumerate(model.ranges):
                        if start<end:
                            self.assertEqual(model.owner(start),cta)
                            self.assertEqual(model.owner(end-1),cta)

    def test_publication_coverage_and_general_bounds(self):
        parent=body(NATIVE,' int g224_parent(')
        row=body(NATIVE,' int g224_row(')
        for stage in range(16):
            pieces=[]
            for unit in range(4*stage,4*stage+4):
                self.assertEqual(expression(re.search(r'return (.*?);',parent)[1],ub=unit),stage)
                for group in range(2):
                    pieces.append((unit,group))
                    self.assertEqual({expression(re.search(r'return (.*?);',row)[1],ub=unit,wg=group,wp=warp)*8+lane for warp in range(4) for lane in range(8)},set(range(unit*64+group*32,unit*64+(group+1)*32)))
            self.assertEqual(len(pieces),8)
        for count in range(1,385):
            jobs=1 if count<=8 else (count+23)//24
            self.assertLessEqual(jobs,1+count/24)
        self.assertLessEqual(32+3072//24,168)
        self.assertLessEqual(3072+7*168,4416)
        self.assertLessEqual(128+2*168,512)
        self.assertLessEqual(512+2*168*16,6144)
        self.assertEqual({(slot*2+group)*1536+tile*512+thread*4+part for slot in (0,264) for group in range(2) for tile in range(3) for thread in range(128) for part in range(4)},set(range(3072))|set(range(264*3072,265*3072)))

    def test_ring_lifetime_helper_completion_and_parity(self):
        segment=body(NATIVE,'__device__ __forceinline__ void wgs_seg(')
        self.assertLess(segment.index('const uint4 cwg = cwn;'),segment.index('mbar_arrive1(S.bars + BW_E + ws)'))
        self.assertLess(segment.index('wg_wait<0>();'),segment.index('mbar_arrive1(S.bars + BX_E + xs)'))
        for signature in ('__device__ void wgs_wproducer(', '__device__ void wgs_xproducer('):
            producer=body(NATIVE,signature)
            self.assertIn('WS_JOIN();',producer)
            self.assertIn('for (int pb = 0; pb < 2; ++pb)',producer)
            self.assertIn('wgs_split(pb, g0, g1)',producer)
        for length in (0,1,2,3,4,15,1024,172032):
            for sequence in range(length):
                slot,parity=sequence%3,(sequence//3)&1
                if sequence>=3:
                    previous=sequence-3
                    self.assertEqual(slot,previous%3)
                    self.assertNotEqual(parity,(previous//3)&1)
        self.assertIn('k < BX_F ? NT : k < BX_E ? 1 : NT',NATIVE)
        kernel=body(NATIVE,'__global__ void __launch_bounds__(NTL, 2) moe_fused(')
        self.assertLess(kernel.index('wgs_phase<true>'),kernel.rindex('WS_JOIN();'))
        self.assertLess(kernel.rindex('WS_JOIN();'),kernel.index('float* red = reinterpret_cast<float*>(xbuf);'))

    def test_deferred_finisher_event_dag_and_numeric_boundaries(self):
        original=gzip.decompress(Path('docs/steering/evidence/2026-11-02-worker-g224-reference-readiness/original-selected.cu.gz').read_bytes()).decode()
        signature='__device__ __forceinline__ void wgs_epi_tile('
        self.assertEqual(body(original,signature),body(NATIVE,signature))
        for counts in ([1],[9,24,1],[24]*128,[1]*32+[24]*112):
            for grid in (2,16,64,132,264):
                model=SourceModel(counts,1,grid)
                dependencies={}
                publications=collections.defaultdict(list)
                for cta in range(grid):
                    segments=model.segments(cta)
                    for index,(unit,start,end) in enumerate(segments):
                        begin=('begin',cta,index)
                        finish=('end',cta,index)
                        publish=('publish',cta,index)
                        dependencies[begin]=set() if index==0 else {('end',cta,index-1)}
                        dependencies[finish]={begin}
                        if index:
                            dependencies[finish].add(('publish',cta,index-1))
                        dependencies[publish]={finish}
                        if index+1<len(segments):
                            dependencies[publish].add(('begin',cta,index+1))
                        publications[unit].append((start,end,publish))
                for parts in publications.values():
                    self.assertEqual(sum(end-start for start,end,_ in parts),16)
                    finisher=next(event for start,_,event in parts if start==0)
                    dependencies[finisher].update(event for _,_,event in parts if event!=finisher)
                successors=collections.defaultdict(list)
                remaining={event:len(pred) for event,pred in dependencies.items()}
                for event,pred in dependencies.items():
                    for predecessor in pred:
                        successors[predecessor].append(event)
                ready=collections.deque(event for event,count in remaining.items() if count==0)
                visited=0
                while ready:
                    event=ready.popleft()
                    visited+=1
                    for successor in successors[event]:
                        remaining[successor]-=1
                        if remaining[successor]==0: ready.append(successor)
                self.assertEqual(visited,len(dependencies))

    def test_actual_prepared_scratch_and_resource_gate_negatives(self):
        import tempfile
        import types
        import g_expert_g224_chunk_phased as treatment
        from test_g_expert_g224_chunk_phased import capture_treatment
        from g224_reference_readiness import gate
        with tempfile.TemporaryDirectory() as directory:
            variant,_,_=treatment.generate(Path(__file__).resolve().parents[1],Path(directory))
        _,env=capture_treatment(variant,True)
        allocations=[]
        def zeros(shape,**kwargs):
            allocations.append((shape,kwargs))
            return shape
        env['torch']=types.SimpleNamespace(zeros=zeros,int32='int32',float32='float32',bfloat16='bfloat16')
        env['_df_inter']=lambda value:value
        env['_MA_EMAX']=3072
        scratch=env['_ma_make_scratch']('cpu-standin')
        self.assertEqual(len(scratch),3)
        self.assertEqual([shape for shape,_ in allocations],[10248,3072+3072*4096+2*265*2*1536+2*265*2+168*16,(3072+2*4416)*4096])
        self.assertTrue(all(args['device']=='cpu-standin' for _,args in allocations))
        normal=dict(stack_bytes=8,spill_store_bytes=8,spill_load_bytes=8,barriers=16,registers=96)
        reference=dict(resources=dict(kernel=normal))
        self.assertTrue(gate(reference,reference)['kernel']['pass'])
        for field in normal:
            worse=dict(normal)
            worse[field]+=1
            self.assertFalse(gate(reference,dict(resources=dict(kernel=worse)))['kernel']['pass'])

    def test_replay_private_state_and_union_handoff(self):
        kernel=body(NATIVE,'__global__ void __launch_bounds__(NTL, 2) moe_fused(')
        self.assertIn('const int o = 1 - S.gen;',kernel)
        self.assertIn('i < 32 + MF_XRS + JMAX * 16',kernel)
        self.assertIn('2u * gridDim.x - 1u',kernel)
        for grid in (2,132,264):
            counter=0
            for launch in range(10):
                generations=[]
                for cta in range(grid):
                    generations.append(int(counter>=grid))
                    counter=(counter+1)%(2*grid)
                self.assertEqual(set(generations),{launch&1})
        route=body(NATIVE,'__device__ void route(')
        producer=body(NATIVE,'__device__ void wgs_xproducer(')
        self.assertTrue(route.rstrip().endswith('CTA_SYNC();\n}'))
        self.assertLess(kernel.index('route(p, S);'),kernel.index('__threadfence_block();'))
        self.assertLess(producer.index('bar.sync 7'),producer.index('S.wcp[0] = 0;'))
        self.assertLess(producer.index('S.wcp[0] = 0;'),producer.index('wgs_ranges(p, S);'))
        self.assertEqual(NATIVE.count('S.wc['),route.count('S.wc['))


class PackagingContracts(unittest.TestCase):
    def test_plain_formatter_keeps_native_string_values(self):
        from g224_reference_pack import pack
        sample='value = "literal # comment stays" # remove\n\nother = """\n\n # literal comment\n"""\n'
        result=pack(sample)
        self.assertIn('literal # comment stays',result)
        self.assertIn('\n\n # literal comment\n',result)
        self.assertNotIn('# remove',result)

    def test_redundant_binding_rejects_changed_literal(self):
        from g224_reference_pack import consolidate
        with self.assertRaises(AssertionError):
            consolidate('_G224_CHUNK_OLD = "not the frozen body"\n')


if __name__=='__main__':
    unittest.main()
