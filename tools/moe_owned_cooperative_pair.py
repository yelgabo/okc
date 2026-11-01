"""Common checked cooperative admission for the two frozen owned MoE launches.

No device-kernel body, memory ordering, precision or PDL mechanism is changed.
The normal owned source has no PDL option on these two wrappers.
"""
import hashlib
from moe_multiply_decode import PIN

TREATMENT_PIN='78383ee6e45baa59c849514f922c768cf3b484449914613f2336e6f5f47f6aa2'
BODY=r'''
    auto kernel = @KERNEL@;
    const int block = @THREADS@, smem = sizeof(@SHARED@);
    int device=0, resident=0;
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch,
                "OWNED_MD_COOP_RESOURCE: requires SM90 cooperative launch");
    TORCH_CHECK(sms>0 && sms<=prop.multiProcessorCount && sms<=136,
                "OWNED_MD_COOP_RESOURCE: grid bounds");
    TORCH_CHECK(p.M>0 && p.M<=kEmax/kTopK && p.lo==0 && p.nloc==32,
                "OWNED_MD_COOP_RESOURCE: chunk/local bounds");
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,block,smem));
    TORCH_CHECK(resident>=1 && sms<=resident*prop.multiProcessorCount,
                "OWNED_MD_COOP_RESOURCE: whole grid cannot reside");
    Params launch_params=p;
    void* args[]={&launch_params};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)kernel,dim3(sms),dim3(block),args,smem,stream));
'''


def wrapper(template,signature,kernel,threads,shared):
    body=BODY.replace('@KERNEL@',kernel).replace('@THREADS@',threads).replace('@SHARED@',shared)
    return template+'\n'+signature+' {'+body+'}'


MAIN=wrapper('template <int MODE>','void launch_main(const Params& p, int sms, cudaStream_t stream)',
             'moe_main_kernel<MODE>','kNTM','SmemMain')
SKINNY=wrapper('template <int SPLIT>','void launch(const Params& p, int sms, cudaStream_t stream)',
               'moe_fused_kernel<SPLIT>','kNTL','Smem')


def cooperative_source(source):
    assert hashlib.sha256(source.encode()).hexdigest() in (PIN,TREATMENT_PIN),'Unreviewed owned pair'
    for anchor,new in (('template <int MODE>\nvoid launch_main(',MAIN),('template <int SPLIT>\nvoid launch(',SKINNY)):
        assert source.count(anchor)==1
        a=source.index(anchor);brace=source.index('{',a);z=brace+1;depth=1
        while depth:depth+=(source[z]=='{')-(source[z]=='}');z+=1
        old=source[a:z]
        assert '<<<' in old and 'cudaLaunchKernelEx' not in old
        source=source[:a]+new+source[z:]
    return source
