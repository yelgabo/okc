"""Read-only host queries for the fixed outgoing-only seam."""
PRIMARY = r'''
std::vector<int64_t> seam_native_identity() {
  return {(int64_t)(uintptr_t)okc::moe_fused<false>,(int64_t)(uintptr_t)okc::moe_fused<true>};
}
'''
SECONDARY = r'''
#include <vector>
std::vector<int64_t> seam_unpool_resources(int64_t pmax) {
  auto kernel=sk2_kernel<1,8,1,1,8,2,4,true,false,2,1,__nv_bfloat16,true,false>;
  cudaFuncAttributes a{};int device=0,optin=0,resident=0;
  const size_t shared=1024+4*28672+16*4+32+4*((16*9+3)&~3)*4+(size_t)pmax*8+128;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,optin-(int)a.sharedSizeBytes));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,416,shared));
  return {(int64_t)(uintptr_t)kernel,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,
          (int64_t)shared,a.maxThreadsPerBlock,resident,416};
}
std::vector<std::vector<int64_t>> seam_graph_edges(int64_t address) {
  cudaGraph_t graph=(cudaGraph_t)(uintptr_t)address;
  size_t count=0;C10_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&count));
  std::vector<cudaGraphNode_t> nodes(count);C10_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&count));
  std::vector<std::vector<int64_t>> rows{{2,cudaGraphDependencyTypeDefault,cudaGraphDependencyTypeProgrammatic,cudaGraphKernelNodePortProgrammatic}};
  for(auto node:nodes) {
    cudaGraphNodeType type{};C10_CUDA_CHECK(cudaGraphNodeGetType(node,&type));
    if(type==cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams params{};C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));
      rows.push_back({0,(int64_t)(uintptr_t)node,(int64_t)(uintptr_t)params.func});
    }
  }
  count=0;C10_CUDA_CHECK(cudaGraphGetEdges(graph,nullptr,nullptr,nullptr,&count));
  std::vector<cudaGraphNode_t> from(count),to(count);std::vector<cudaGraphEdgeData> edges(count);
  C10_CUDA_CHECK(cudaGraphGetEdges(graph,from.data(),to.data(),edges.data(),&count));
  for(size_t index=0;index<count;++index) {
    for(auto byte:edges[index].reserved) TORCH_CHECK(byte==0,"seam unknown edge ABI");
    rows.push_back({1,(int64_t)(uintptr_t)from[index],(int64_t)(uintptr_t)to[index],edges[index].from_port,edges[index].to_port,edges[index].type});
  }
  return rows;
}
'''
DECL = '\n#include <vector>\nstd::vector<int64_t> seam_unpool_resources(int64_t pmax);\nstd::vector<std::vector<int64_t>> seam_graph_edges(int64_t address);\n'
