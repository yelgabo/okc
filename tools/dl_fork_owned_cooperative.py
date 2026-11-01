"""Optional future owned-expert launch contract. No change to current G1a/G1b."""
import ast
from dl_fork_owned_moe import module_source

LAUNCH=r'''template <typename Kern>
void moe1_launch(Kern kern, int grid, int block, int smem, cudaStream_t stream,
                 const Params& p) {
    TORCH_CHECK(!p.pdl, "OWNED_COOP_RESOURCE: PDL overlap unsupported");
    int device=0, resident=0;
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch,
                "OWNED_COOP_RESOURCE: requires SM90 cooperative launch");
    TORCH_CHECK(grid>0 && grid<=prop.multiProcessorCount && block>0 && smem>=0,
                "OWNED_COOP_RESOURCE: grid/block/shared bounds");
    TORCH_CHECK(p.M>0 && p.M<=kEmax/kTopK && p.nloc>=1 && p.nloc<=kLocalMax,
                "OWNED_COOP_RESOURCE: chunk/local expert bounds");
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kern,block,smem));
    TORCH_CHECK(resident>=1 && grid<=resident*prop.multiProcessorCount,
                "OWNED_COOP_RESOURCE: whole grid cannot reside");
    Params launch_params=p;
    void* args[]={&launch_params};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)kern,dim3(grid),dim3(block),
                                              args,smem,stream));
}
'''

def cooperative_module_source(owned):
    """Same two kernels/ABI; replace only their shared host launch helper."""
    original=module_source(owned)
    tree=ast.parse(original)
    node=next(n for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='_OWNED_MOE_DEVICE' for t in n.targets))
    device=ast.literal_eval(node.value)
    a=device.index('template <typename Kern>\nvoid moe1_launch(')
    z=device.index('\n}\n',a)+3
    old=device[a:z]
    assert 'kern<<<grid, block, smem, stream>>>(p);' in old and 'cudaLaunchKernelEx' in old
    changed=device[:a]+LAUNCH+device[z:]
    assert changed.replace(LAUNCH,old,1)==device
    literal=ast.get_source_segment(original,node.value)
    assert original.count(literal)==1
    result=original.replace(literal,repr(changed),1)
    assert result.count("name='okc_fork_owned_moe_v1'")==1
    result=result.replace("name='okc_fork_owned_moe_v1'","name='okc_fork_owned_moe_coop_v1'")
    ast.parse(result)
    return result
