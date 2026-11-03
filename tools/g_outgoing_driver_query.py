"""Unintegrated read-only CUDA 13 driver-identity bridge; never dispatches."""
import json

from g_outgoing_pair_read import edge_gate


IDENTITY_NAMESPACE = "CUDA13:CUfunction"
QUERY_EXTRA_LDFLAGS = ("-lcuda",)
HOST_COMMON = r'''
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <vector>
static void seam_driver_check(int code, const char* api, int64_t node, int64_t row, int64_t arm) {
  if (code != 0) {
    std::fprintf(stderr,"SEAM_DRIVER_ERROR api=%s code=%d node=%lld row=%lld arm=%lld\n",api,code,(long long)node,(long long)row,(long long)arm);
    std::fflush(stderr);
    TORCH_CHECK(false,"SEAM_DRIVER_ERROR api=",api," code=",code," node=",node," row=",row," arm=",arm);
  }
}
static int64_t seam_driver_symbol(const void* symbol, const char* label, int64_t row, int64_t arm) {
  cudaFunction_t function{};
  seam_driver_check((int)cudaGetFuncBySymbol(&function,symbol),label,0,row,arm);
  seam_driver_check(function ? 0 : -1,"cudaGetFuncBySymbol:null",0,row,arm);
  return (int64_t)(uintptr_t)reinterpret_cast<CUfunction>(function);
}
'''
PRIMARY = HOST_COMMON + r'''
std::vector<int64_t> seam_driver_primary_identity(int64_t row, int64_t arm) {
  return {seam_driver_symbol((const void*)okc::moe_fused<false>,"cudaGetFuncBySymbol:primary:false",row,arm),
          seam_driver_symbol((const void*)okc::moe_fused<true>,"cudaGetFuncBySymbol:primary:true",row,arm)};
}
'''
SECONDARY = HOST_COMMON + r'''
int64_t seam_driver_unpool_identity(int64_t row, int64_t arm) {
  return seam_driver_symbol((const void*)sk2_kernel<1,8,1,1,8,2,4,true,false,2,1,__nv_bfloat16,true,false>,
                            "cudaGetFuncBySymbol:unpool",row,arm);
}
std::vector<std::vector<int64_t>> seam_driver_graph_edges(int64_t address, int64_t row, int64_t arm) {
  CUgraph graph=(CUgraph)(uintptr_t)address;
  size_t count=0;
  seam_driver_check((int)cuGraphGetNodes(graph,nullptr,&count),"cuGraphGetNodes:count",0,row,arm);
  const size_t capacity=count;
  std::vector<CUgraphNode> nodes(count);
  seam_driver_check((int)cuGraphGetNodes(graph,nodes.data(),&count),"cuGraphGetNodes:fill",0,row,arm);
  seam_driver_check(count==capacity ? 0 : -1,"cuGraphGetNodes:count_changed",0,row,arm);
  std::vector<std::vector<int64_t>> rows{{2,CU_GRAPH_DEPENDENCY_TYPE_DEFAULT,CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC,CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC}};
  for(auto node:nodes) {
    const int64_t identity=(int64_t)(uintptr_t)node;
    CUgraphNodeType type{};
    seam_driver_check((int)cuGraphNodeGetType(node,&type),"cuGraphNodeGetType",identity,row,arm);
    if(type==CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS params{};
      seam_driver_check((int)cuGraphKernelNodeGetParams(node,&params),"cuGraphKernelNodeGetParams_v2",identity,row,arm);
      seam_driver_check(params.func ? 0 : -1,"cuGraphKernelNodeGetParams_v2:non_CUfunction_representation",identity,row,arm);
      rows.push_back({0,identity,(int64_t)(uintptr_t)params.func});
    }
  }
  count=0;
  seam_driver_check((int)cuGraphGetEdges(graph,nullptr,nullptr,nullptr,&count),"cuGraphGetEdges_v2:count",0,row,arm);
  const size_t edge_capacity=count;
  std::vector<CUgraphNode> from(count),to(count);
  std::vector<CUgraphEdgeData> edges(count ? count : 1);
  seam_driver_check((int)cuGraphGetEdges(graph,from.data(),to.data(),edges.data(),&count),"cuGraphGetEdges_v2:fill",0,row,arm);
  seam_driver_check(count==edge_capacity ? 0 : -1,"cuGraphGetEdges_v2:count_changed",0,row,arm);
  for(size_t index=0;index<count;++index) {
    for(auto byte:edges[index].reserved)
      seam_driver_check(byte==0 ? 0 : -1,"cuGraphGetEdges_v2:reserved",(int64_t)(uintptr_t)from[index],row,arm);
    rows.push_back({1,(int64_t)(uintptr_t)from[index],(int64_t)(uintptr_t)to[index],edges[index].from_port,edges[index].to_port,edges[index].type});
  }
  return rows;
}
'''
PRIMARY_DECL = "std::vector<int64_t> seam_driver_primary_identity(int64_t row, int64_t arm);"
SECONDARY_DECL = """
int64_t seam_driver_unpool_identity(int64_t row, int64_t arm);
std::vector<std::vector<int64_t>> seam_driver_graph_edges(int64_t address, int64_t row, int64_t arm);
"""


def validate_edges(rows, primary, consumer, enabled):
    assert primary["namespace"] == consumer["namespace"] == IDENTITY_NAMESPACE, "DRIVER_IDENTITY_NAMESPACE"
    values = primary["values"]
    target = consumer["values"]
    assert len(values) == len(set(values)) == 2 and len(target) == 1, "DRIVER_IDENTITY_ARITY"
    assert all(type(value) is int and value > 0 for value in values + target), "DRIVER_IDENTITY_VALUE"
    assert target[0] not in values, "DRIVER_IDENTITY_COLLISION"
    return edge_gate(rows, values, target[0], enabled)


def inspect_capture(graph, primary_extension, query_extension, *, row, arm, enabled, resources, build_census, emit=print):
    assert type(row) is int and row >= 0 and type(arm) is int and arm in range(4), "DRIVER_CONTEXT"
    assert type(enabled) is bool and enabled == (arm == 3), "DRIVER_ARM_POLICY"
    assert resources and build_census, "DRIVER_PREFIX_REQUIRED"
    prefix = dict(row=row, arm=arm, identity_namespace=IDENTITY_NAMESPACE, resources=resources, build_census=build_census)
    emit("SEAM_DRIVER_PREFIX " + json.dumps(prefix, sort_keys=True, allow_nan=False), flush=True)
    api = "seam_driver_primary_identity"
    try:
        primary = dict(namespace=IDENTITY_NAMESPACE, values=list(primary_extension.seam_driver_primary_identity(row, arm)))
        api = "seam_driver_unpool_identity"
        consumer = dict(namespace=IDENTITY_NAMESPACE, values=[query_extension.seam_driver_unpool_identity(row, arm)])
        api = "seam_driver_graph_edges"
        rows = list(query_extension.seam_driver_graph_edges(graph.raw_cuda_graph(), row, arm))
        api = "validate_edges"
        proof = validate_edges(rows, primary, consumer, enabled)
        return dict(identity_namespace=IDENTITY_NAMESPACE, primary=primary, consumer=consumer, rows=rows, proof=proof)
    except Exception as error:
        emit("SEAM_DRIVER_FAILURE " + json.dumps(dict(row=row, arm=arm, api=api, error=str(error)), sort_keys=True), flush=True)
        raise
