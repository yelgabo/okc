"""Executable generated-formula/real-factory contracts; CPU only, not GPU proof."""
import ast
import collections
import json
from pathlib import Path
import re
import tempfile
import types
import unittest

import g_expert_g224_prefix as generator
from g_native_patch_ladder import flags
from prod_size_compile_g import bindings

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / generator.BASE).read_text(encoding="utf8")
NATIVE = generator.transform_native(generator.literal(BASE, "MOE_CUDA_SRC"))


def body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth, cursor = 1, opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


def expression(text, **values):
    text = re.sub(r"\((?:unsigned long long|unsigned|uint32_t|size_t|int)\)", "", text)
    text = text.replace("(pb ? 16 : 32)", "(16 if pb else 32)").replace("(pb ? 5 : 6)", "(5 if pb else 6)")
    text = re.sub(r"(\d+)[uU](?:ll)?\b", r"\1", text)
    text = text.replace(" / ", " // ")
    return eval(text, {"__builtins__": {}, "min": min, "max": max}, values)


def helper(name, **values):
    source = body(NATIVE, " " + name + "(")
    return expression(re.search(r"return (.*?);", source, re.S)[1], **values)


def capture(source, modified):
    env = bindings(BASE)
    env.update(flags(BASE))
    calls = []
    class Extension:
        def okc_set_pdl(self, value):
            assert value == 0
    def load_inline(**kwargs):
        calls.append(kwargs)
        return Extension()
    env.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
               load_inline=load_inline, _MA_EXTS={}, _MA_MAIN_NS=3, _MA_PDL=False,
               _MA_ALT_SRCS=bindings(BASE)["_MA_ALT_SRCS"], MOE_CUDA_SRC=generator.literal(BASE,"MOE_CUDA_SRC"),
               MOE_CPP_SRC=generator.literal(BASE,"MOE_CPP_SRC"), _RouterExt=lambda value: value)
    for name in ("_main_defines", "_ma_ext", "_fork_native_moe"):
        exec(generator.function_source(BASE, name), env)
    if modified:
        env["_G224_CUDA"] = NATIVE
        exec(generator.APPENDIX, env)
    state = dict(ma_sfx=("-DOKC_PRMTSC=1",), ma_sc6=True, ma_ext={})
    for tokens in (1, 8, 16, 32, 128, 256, 384):
        env["_fork_native_moe"](tokens, state)
    return calls, env


def source_contract(source):
    required = ('#define OKC_FT_T 24', 'NW == 8 && NT == 256 && NTL == 320 && WGS_NT == 3',
        '__launch_bounds__(NTL, 2) moe_fused', 'WGS_SLOTS = 265', 'MF_XRS = JMAX, MF_DA = 512, MF_INTS = 6144',
        'sizeof(Smem) == 23096', '== 115328, "G224 dynamic layout"',
        'union { int wc[NW][32]; int wcp[JMAX + 1]; };', 'resident>=2',
        'g224_weight_byte(pb, S.jE[j], ub, st, 0)', 'r16 = g224_row(ub, wg, wp)',
        '((ub & 1) * 2 + wg) * WGS_PIECE', 'return ub >> 2;', 'return ub * 8 + wg * 4 + wp;',
        'job << (10 - pb)', 'stage >> (10 - pb)', '((unsigned long long)total * (unsigned)cta',
        'column = tid + chunk * NT', 'for (int w = 0; w < 2 * NW; ++w)',
        'const int virtual_warp = warp + slice * NW', 'constexpr int SEG = 512',
        '(grid *= 2)>0 && grid==2*prop.multiProcessorCount')
    for token in required:
        assert token in source, token
    route = body(source, '__device__ void route(')
    assert 'if (tid == 0) S.wc[0][0] = 0;\n  CTA_SYNC();' in route
    assert route.rstrip().endswith('CTA_SYNC();\n}')
    assert source.count('S.wc[') == route.count('S.wc[')
    assert 'S.wcp' not in route
    producer = body(source, '__device__ void wgs_xproducer(')
    assert all(token in producer for token in ('bar.sync 7', 'S.wcp[0] = 0', '__syncwarp()', 'wgs_ranges(p, S);'))
    assert producer.index('bar.sync 7') < producer.index('S.wcp[0] = 0') < producer.index('__syncwarp()') < producer.index('wgs_ranges(p, S);')
    kernel = body(source, '__global__ void __launch_bounds__(NTL, 2) moe_fused(')
    assert kernel.index('route(p, S);') < kernel.index('__threadfence_block();') < kernel.index('bar.arrive 7')
    assert source.count('wgs_ranges(p, S);') == 1
    assert '#if FAST1_ON || OKC_WGS\n  return;' in body(source, '__device__ void early_issue(const Params& p, Smem& S) {')
    assert 'wgs_prefix(S,' not in source
    segment = body(source, '__device__ __forceinline__ void wgs_seg(')
    assert segment.index('wg_wait<0>();') < segment.index('mbar_arrive1(S.bars + BX_E + xs)')
    assert 'fence.proxy.async.shared::cta;' in segment
    assert source.count('g224_parent(ub)),') == 2
    assert 'wgs_seg<PB, 4>' not in source and 'struct WGSMMA<4>' not in source
    assert 'WGS_NS = 3' in source and 'OKC_WGSNX 3' in source
    assert 'mf_gen(S) * MF_XRS' in source and 'o * (JMAX * 16)' in source
    return True


class GeneratedContracts(unittest.TestCase):
    def test_generation_and_factory_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            variant, native, manifest = generator.generate(ROOT, Path(folder))
        self.assertTrue(variant.startswith(BASE)); self.assertEqual(native, NATIVE)
        self.assertTrue(source_contract(native))
        original, _ = capture(BASE, False)
        adapted, env = capture(variant, True)
        self.assertEqual(len(original), len(adapted))
        long_count = 0
        for old, new in zip(original, adapted):
            self.assertEqual(old['cpp_sources'], new['cpp_sources'])
            expected = ['-DOKC_NW=8' if item == '-DOKC_NW=16' else item for item in old['extra_cuda_cflags']]
            if '-DOKC_WGS=1' in old['extra_cuda_cflags']:
                long_count += 1
                self.assertEqual(new['extra_cuda_cflags'], expected)
                self.assertEqual(new['cuda_sources'], native)
            else:
                for key in ('cuda_sources', 'extra_cuda_cflags', 'functions'):
                    self.assertEqual(old[key], new[key])
        self.assertEqual(long_count, 2)
        sizes = []
        env.update(torch=types.SimpleNamespace(int32='i', float32='f', bfloat16='b', zeros=lambda count, **kwargs: sizes.append((count,kwargs)) or count), _MA_EMAX=3072, _df_inter=lambda value: value)
        env['_ma_make_scratch']('cpu-stub')
        self.assertEqual([item[0] for item in sizes], [8+1024+3072+6144,3072+3072*4096+2*265*2*1536+2*265*2+168*16,(3072+2*4416)*4096])

    def test_packed_piece_bijection(self):
        for piece in (8960, 9216):
            for pb in (0, 1):
                units = 64 >> pb
                for expert in range(32):
                    actual = set()
                    for unit in range(units):
                        for stage in range(16):
                            for wg in range(2):
                                address = helper('g224_weight_byte', pb=pb,expert=expert,ub=unit,stage=stage,wg=wg,WGS_PIECE=piece)
                                expected = ((expert*(units//2)+unit//2)*16+stage)*4*piece+((unit%2)*2+wg)*piece
                                self.assertEqual(address, expected)
                                actual.add(address)
                    expected = {((expert*(units//2)+unit)*16+stage)*4*piece+wg*piece for unit in range(units//2) for stage in range(16) for wg in range(4)}
                    self.assertEqual(actual,expected)

    def test_gather_and_output_bijections(self):
        gather = body(NATIVE, '__device__ void wgs_gather(')
        index = re.search(r'\+ (\(+column >> 5.*?);', gather)[1]
        for count in range(1,25):
            tiles = (count+7)//8
            offsets=set()
            for row in range(count):
                for tid in range(256):
                    for chunk in range(2):
                        column=tid+chunk*256
                        offset=expression(index,column=column,ntt=tiles,r=row)
                        self.assertEqual(offset, ((((column>>5)*tiles+(row>>3))*32+(column&31))*8+(row&7)))
                        self.assertLess(offset,tiles*8*512)
                        offsets.add(offset)
            self.assertEqual(len(offsets),count*512)
        self.assertEqual({(tid+slice*256)*8+column for tid in range(256) for slice in range(2) for column in range(8)},set(range(4096)))
        self.assertEqual({segment*512+tid*2+column for segment in range(8) for tid in range(256) for column in range(2)},set(range(4096)))
        self.assertEqual({(warp+slice*8)//2 for warp in range(8) for slice in range(2)},set(range(8)))

    def test_jobs_bounds_and_publications(self):
        route = body(NATIVE,'__device__ void route(')
        self.assertIn('(tot + FT_T - 1) / FT_T',route)
        self.assertIn('S.jC[fj + q] = min(FT_T, tot - q * FT_T)',route)
        count_expr = re.search(r'S.jC\[fj \+ q\] = (.*?);',route)[1]
        for tokens in (0,1,8,9,23,24,25,32,128,256,384):
            for mode in ('empty','one','full','spread'):
                rows=[tuple(range(32,40)) if mode=='empty' else tuple(range(8)) if mode=='full' else tuple((token*8+slot)%32 for slot in range(8)) if mode=='spread' else (0,32,33,34,35,36,37,38) for token in range(tokens)]
                self.assertTrue(all(len(set(row))==8 for row in rows))
                counts=collections.Counter(expert for row in rows for expert in row if expert<32)
                jobs=[]
                for expert, total in sorted(counts.items()):
                    self.assertLessEqual(total,384)
                    if total<=8: jobs.append((expert,total))
                    else:
                        for unit in range((total+23)//24):
                            jobs.append((expert,expression(count_expr,FT_T=24,tot=total,q=unit)))
                self.assertLessEqual(len(jobs),168)
                self.assertEqual(sum(count for _,count in jobs),sum(counts.values()))
                self.assertLessEqual(sum((count+7)//8*8 for _,count in jobs),4416)
        for parent in range(16):
            contributors=[]
            columns=set()
            for unit in range(parent*4,parent*4+4):
                self.assertEqual(helper('g224_parent',ub=unit),parent)
                for wg in range(2):
                    contributors.append((unit,wg))
                    for wp in range(4):
                        row=helper('g224_row',ub=unit,wg=wg,wp=wp)
                        columns.update(range(row*8,row*8+8))
            self.assertEqual(len(contributors),8)
            self.assertEqual(columns,set(range(parent*256,parent*256+256)))
        self.assertLessEqual(128+2*168,512)
        self.assertLessEqual(512+2*168*16,6144)
        self.assertGreater(265,264)

    def test_lifetime_and_geometry_negatives(self):
        self.assertTrue(source_contract(NATIVE))
        for before, after in (('WGS_SLOTS = 265','WGS_SLOTS = 160'),('MF_XRS = JMAX','MF_XRS = 96'),
            ('return ub >> 2;','return ub >> 1;'),('return ub * 8 + wg * 4 + wp;','return ub * 16 + wg * 4 + wp;'),
            ('bar.sync 7','bar.sync 6'),('if (tid == 0) S.wc[0][0] = 0;\n  CTA_SYNC();','if (tid == 0) S.wc[0][0] = 0;'),
            ('S.wcp[0] = 0;\n  __syncwarp();','S.wcp[0] = 0;'),('wgs_ranges(p, S);','wgs_ranges(p, S); int late = S.wc[0][0];'),
            ('o * (JMAX * 16)','o * (136 * 16)')):
            with self.assertRaises(AssertionError,msg=before): source_contract(NATIVE.replace(before,after,1))
        with self.assertRaises(AssertionError):
            invalid=(0,0,1,2,3,4,5,6)
            assert len(set(invalid))==8


if __name__ == '__main__':
    unittest.main()
