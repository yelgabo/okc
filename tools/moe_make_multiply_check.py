"""Four independently packed owned arms, one decoder factor, CHECK ONLY."""
import ast
from pathlib import Path
import sys
from moe_multiply_decode import patch
from moe_owned_cooperative_pair import cooperative_source

RESOURCE=r'''
std::vector<int64_t> multiply_resources() {
    std::vector<int64_t> out;
    auto query = [&](auto kernel,int tag,int threads,int sm) {
        cudaFuncAttributes a{}; int blocks=0;
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));
        C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,sm));
        out.insert(out.end(),{tag,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,sm,blocks,threads,a.maxThreadsPerBlock});
    };
    query(sk_8_3_0::moe_fused_kernel<1>,0,sk_8_3_0::kNTL,sizeof(sk_8_3_0::Smem));
    query(moe_main_kernel<47>,1,kNTM,sizeof(SmemMain));
    return out;
}
'''


def limited(source):
    # Compile only kernels selected by the exact published abe765 policy.
    # No preparation self-test/tuner is invoked; harness runs its own FP32 gates.
    assert source.count('name="okc_moe_fused",')==1
    source=source.replace('name="okc_moe_fused",','name=_MD_MODULE_NAME,')
    source=source.replace('int64_t moe_smem_bytes();','int64_t moe_smem_bytes();\nstd::vector<int64_t> multiply_resources();',1)
    assert source.count('int64_t moe_smem_bytes() {')==1
    source=source.replace('int64_t moe_smem_bytes() {',RESOURCE+'\nint64_t moe_smem_bytes() {',1)
    source=source.replace('functions=["moe_run", "moe_smem_bytes"],','functions=["moe_run", "moe_smem_bytes", "multiply_resources"],',1)
    ast.parse(source)
    return source


def build(source,body):
    altered=patch(source)
    out='# okc: problem=muse_moe_w4a16\n# CHECK ONLY: fully owned lossless multiply decoder vs frozen owned control.\n'
    out+='import torch\nimport time\n'
    out+='_MD_CONTROL='+repr(limited(cooperative_source(source)))+'\n'
    out+='_MD_CANDIDATE='+repr(limited(cooperative_source(altered)))+'\n'
    out+=body
    ast.parse(out)
    assert len(out.encode())<1_000_000
    return out


if __name__=='__main__':
    import hashlib
    root=Path(__file__).resolve().parents[1]
    src,dst=map(Path,sys.argv[1:])
    result=build(src.read_text(encoding='utf-8'),(root/'tools/moe_multiply_check_body.py').read_text(encoding='utf-8'))
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(result,encoding='utf-8',newline='\n')
    print('MULTIPLY_CHECK',len(result.encode()),hashlib.sha256(result.encode()).hexdigest())
