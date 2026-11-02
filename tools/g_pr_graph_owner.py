"""One scratch PR graph-owner source transform; never runs CUDA locally."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
BASE_SHA = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
EVIDENCE = ROOT / 'docs/steering/evidence/2026-11-02-worker-pr-graph-owner'

HOST = r'''
#include <torch/extension.h>
#include <ATen/ops/from_blob.h>
#include <cuda.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace py = pybind11;
static_assert(std::atomic<bool>::is_always_lock_free, "callback requires a lock-free signal");

static void require(CUresult result, const char* stage) {
  if (result != CUDA_SUCCESS) throw std::runtime_error(std::string(stage) + ":" + std::to_string(result));
}

static void fatal(CUresult result, const char* stage) noexcept {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "PR_GRAPH_FATAL %s %d; refusing unsafe cleanup\n", stage, int(result));
    std::abort();
  }
}

struct Mapping {
  CUdevice device{};
  CUcontext context{};
  CUdeviceptr address{};
  CUmemGenericAllocationHandle handle{};
  CUuserObject user{};
  size_t bytes{};
  bool retained=false, reserved=false, created=false, mapped=false, copy_submitted=false;
  std::atomic<bool> callback_finished{false};

  void cleanup() noexcept {
    if (context) {
      fatal(cuCtxPushCurrent(context), "cleanup-context");
      if (copy_submitted) fatal(cuCtxSynchronize(), "completion");
      if (mapped) fatal(cuMemUnmap(address, bytes), "unmap");
      if (created) fatal(cuMemRelease(handle), "release");
      if (reserved) fatal(cuMemAddressFree(address, bytes), "address-free");
      CUcontext popped{};
      fatal(cuCtxPopCurrent(&popped), "cleanup-pop");
    }
    if (retained) fatal(cuDevicePrimaryCtxRelease(device), "primary-release");
  }
};

static void CUDA_CB graph_release(void* opaque) noexcept {
  static_cast<Mapping*>(opaque)->callback_finished.store(true, std::memory_order_release);
}

static void retirement(Mapping* mapping) noexcept {
  while (!mapping->callback_finished.load(std::memory_order_acquire))
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  mapping->cleanup();
  delete mapping;
}

struct Allocation {
  torch::Tensor tensor;
  Mapping* mapping;
  Allocation(torch::Tensor value, Mapping* owner): tensor(std::move(value)), mapping(owner) {}

  void before_read(uint64_t stream_bits) {
    CUcontext current{};
    require(cuCtxGetCurrent(&current), "read-context");
    if (current != mapping->context) throw std::runtime_error("read-context-mismatch");
    CUstreamCaptureStatus status{};
    cuuint64_t capture_id{};
    CUgraph graph{};
    require(cuStreamGetCaptureInfo(reinterpret_cast<CUstream>(stream_bits), &status,
            &capture_id, &graph, nullptr, nullptr, nullptr), "capture-info");
    if (status == CU_STREAM_CAPTURE_STATUS_INVALIDATED)
      throw std::runtime_error("invalidated-capture");
    if (status != CU_STREAM_CAPTURE_STATUS_NONE && status != CU_STREAM_CAPTURE_STATUS_ACTIVE)
      throw std::runtime_error("unknown-capture-status");
    if (status == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
      if (!graph) throw std::runtime_error("missing-active-graph");
      require(cuGraphRetainUserObject(graph, mapping->user, 1, 0), "graph-retain");
    }
  }
};

static py::tuple allocate(torch::Tensor source, uint64_t stream_bits) {
  std::unique_ptr<Mapping> owner(new Mapping);
  bool user_created=false, worker_started=false;
  try {
    if (!source.is_cuda() || source.scalar_type()!=at::kBFloat16 || !source.is_contiguous()
        || source.sizes()!=at::IntArrayRef({4608,8192}))
      throw std::runtime_error("unsupported-pr-layout");
    CUstream stream=reinterpret_cast<CUstream>(stream_bits);
    CUstreamCaptureStatus capture{};
    require(cuStreamIsCapturing(stream, &capture), "prepare-capture");
    if (capture!=CU_STREAM_CAPTURE_STATUS_NONE) throw std::runtime_error("prepare-during-capture");
    require(cuCtxGetCurrent(&owner->context), "prepare-context");
    if (!owner->context) throw std::runtime_error("missing-current-context");
    require(cuDeviceGet(&owner->device, source.get_device()), "source-device");
    CUcontext primary{};
    require(cuDevicePrimaryCtxRetain(&primary, owner->device), "primary-retain");
    owner->retained=true;
    if (primary!=owner->context) throw std::runtime_error("unsupported-nonprimary-context");
    int vmm=0, compression=0;
    require(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                owner->device), "vmm-support");
    require(cuDeviceGetAttribute(&compression, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
                                owner->device), "compression-support");
    if (!vmm || !compression) throw std::runtime_error("unsupported-capability");
    CUmemAllocationProp requested{};
    requested.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    requested.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
    requested.location.id=owner->device;
    requested.allocFlags.compressionType=CU_MEM_ALLOCATION_COMP_GENERIC;
    size_t granularity=0;
    require(cuMemGetAllocationGranularity(&granularity, &requested, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            "granularity");
    const size_t count=static_cast<size_t>(source.numel());
    if (!granularity || count>std::numeric_limits<size_t>::max()/2)
      throw std::runtime_error("size-overflow");
    const size_t logical=count*2;
    if (logical>std::numeric_limits<size_t>::max()-(granularity-1))
      throw std::runtime_error("rounding-overflow");
    owner->bytes=((logical+granularity-1)/granularity)*granularity;
    require(cuMemAddressReserve(&owner->address, owner->bytes, granularity, 0, 0), "reserve");
    owner->reserved=true;
    require(cuMemCreate(&owner->handle, owner->bytes, &requested, 0), "create");
    owner->created=true;
    CUmemAllocationProp actual{};
    require(cuMemGetAllocationPropertiesFromHandle(&actual, owner->handle), "actual-properties");
    if (actual.allocFlags.compressionType!=CU_MEM_ALLOCATION_COMP_GENERIC
        || actual.type!=requested.type || actual.location.type!=requested.location.type
        || actual.location.id!=requested.location.id)
      throw std::runtime_error("nongeneric-or-wrong-backing");
    require(cuMemMap(owner->address, owner->bytes, 0, owner->handle, 0), "map");
    owner->mapped=true;
    CUmemAccessDesc access{};
    access.location=requested.location;
    access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    require(cuMemSetAccess(owner->address, owner->bytes, &access, 1), "access");
    if (owner->address%128) throw std::runtime_error("alignment");
    owner->copy_submitted=true;
    require(cuMemcpyDtoDAsync(owner->address,
            reinterpret_cast<CUdeviceptr>(source.data_ptr()), logical, stream), "bit-copy");
    require(cuStreamSynchronize(stream), "copy-completion");
    require(cuUserObjectCreate(&owner->user, owner.get(), graph_release, 1,
                              CU_USER_OBJECT_NO_DESTRUCTOR_SYNC), "user-create");
    user_created=true;
    Mapping* mapping=owner.get();
    std::thread worker(retirement, mapping);
    worker.detach();
    worker_started=true;
    owner.release();
    auto tensor=at::from_blob(reinterpret_cast<void*>(mapping->address), source.sizes(), source.strides(),
      [mapping](void*) noexcept { fatal(cuUserObjectRelease(mapping->user, 1), "tensor-user-release"); },
      source.options());
    return py::make_tuple(std::make_shared<Allocation>(std::move(tensor), mapping), "generic-owned");
  } catch (const std::exception& error) {
    if (owner) {
      if (user_created) {
        fatal(cuUserObjectRelease(owner->user, 1), "setup-user-release");
        while (!owner->callback_finished.load(std::memory_order_acquire))
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      owner->cleanup();
    } else if (worker_started) {
      std::fprintf(stderr, "PR_GRAPH_FATAL tensor wrapping after worker publication: %s\n", error.what());
      std::abort();
    }
    return py::make_tuple(py::none(), std::string("no-treatment:")+error.what());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<Allocation, std::shared_ptr<Allocation>>(module, "Allocation")
    .def_property_readonly("tensor", [](Allocation& owner) { return owner.tensor; })
    .def("before_read", &Allocation::before_read);
  module.def("allocate", &allocate);
}
'''

APPEND = '''

_PR_GRAPH_MODULE = None
_PR_GRAPH_HOST = r"""{host}"""

def _pr_graph_module():
    global _PR_GRAPH_MODULE
    if _PR_GRAPH_MODULE is None:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None:
            raise RuntimeError("PR graph-owner CUDA headers unavailable")
        _PR_GRAPH_MODULE = load_inline(name="pr_graph_owner_v1", cpp_sources=_PR_GRAPH_HOST,
            extra_include_paths=[CUDA_HOME + "/include"],
            functions=None, extra_cflags=["-O2", "-std=c++17", "-pthread"],
            extra_ldflags=["-lcuda", "-pthread", "-Wl,-z,nodelete"], with_cuda=False, verbose=False)
    return _PR_GRAPH_MODULE

def _pr_graph_prepare_owner(state):
    owner, status = _pr_graph_module().allocate(state["pr"], torch.cuda.current_stream().cuda_stream)
    state["_pr_graph_status"] = status
    return owner

def _pr_graph_before_read(plan, stream):
    owner = plan.get("_pr_graph_owner")
    if owner is not None:
        weight, expected = plan["weight"], owner.tensor
        if (weight.data_ptr() != expected.data_ptr() or weight.dtype != expected.dtype
                or weight.shape != expected.shape or weight.device != expected.device
                or weight.stride() != expected.stride()):
            raise RuntimeError("PR graph-owner/weight identity mismatch")
        owner.before_read(stream)

_pr_graph_original_deepcopy = _GraftPlan.__deepcopy__
def _pr_graph_plan_deepcopy(self, memo):
    import copy
    owner = self.get("_pr_graph_owner")
    if owner is None:
        return _pr_graph_original_deepcopy(self, memo)
    weight, expected = self["weight"], owner.tensor
    if (weight.data_ptr() != expected.data_ptr() or weight.dtype != expected.dtype
            or weight.shape != expected.shape or weight.device != expected.device
            or weight.stride() != expected.stride()):
        raise RuntimeError("PR immutable owner tag mismatch")
    result = type(self)()
    memo[id(self)] = result
    for key, value in self.items():
        result[key] = value if key in ("layouts", "weight", "_pr_graph_owner") else copy.deepcopy(value, memo)
    return result
_GraftPlan.__deepcopy__ = _pr_graph_plan_deepcopy
'''


def function(source, name):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes)!=1:
        raise ValueError(name)
    return ast.get_source_segment(source, nodes[0])


def transforms(original):
    plan = function(original, '_graft_lt_plan')
    changed = plan.replace('    src = {"pr": st["pr"],',
        '    pr_owner = _pr_graph_prepare_owner(st)\n    src = {"pr": pr_owner.tensor if pr_owner is not None else st["pr"],', 1)
    needle = '        plans[(name, m)] = _GraftPlan(lt=lt, layouts=layouts, algo=algo, weight=weight, n=n, fp32=fp32)'
    changed = changed.replace(needle, needle+'\n        if name == "pr" and pr_owner is not None:\n            plans[(name, m)]["_pr_graph_owner"] = pr_owner', 1)
    changed = changed.replace('    st["graft_lt"] = plans',
        '    if pr_owner is not None and not any(name == "pr" for name, tokens in plans):\n        st["_pr_graph_status"] = "no-treatment:no-accepted-pr-plan"\n    st["graft_lt"] = plans', 1)
    run = function(original, '_graft_lt_run')
    changed_run = run.replace('    lt.call("cublasLtMatmul",',
        '    stream = torch.cuda.current_stream().cuda_stream\n    _pr_graph_before_read(plan, stream)\n    lt.call("cublasLtMatmul",', 1)
    changed_run = changed_run.replace('_GP(torch.cuda.current_stream().cuda_stream))', '_GP(stream))', 1)
    return [(plan, changed), (run, changed_run)]


def build(original):
    payload = original.encode()
    if len(payload)!=869076 or hashlib.sha256(payload).hexdigest()!=BASE_SHA:
        raise ValueError('not exact ba9 base')
    source = original
    for old, new in transforms(original):
        if source.count(old)!=1 or old==new:
            raise ValueError('nonunique transform')
        source = source.replace(old, new, 1)
    return source + APPEND.format(host=HOST)


def validate(source, original):
    if source!=build(original):
        raise ValueError('source differs from the one frozen transform')
    if len(source.encode())>=1000000 or '\r' in source or not source.endswith('\n'):
        raise ValueError('physical packaging failure')
    ast.parse(source)
    reversed_source = source[:-len(APPEND.format(host=HOST))]
    for old, new in reversed(transforms(original)):
        reversed_source = reversed_source.replace(new, old, 1)
    if reversed_source != original:
        raise ValueError('exact reversal failed')
    return dict(bytes=len(source.encode()), sha256=hashlib.sha256(source.encode()).hexdigest(),
                exact_reversal=True, native_sources_unchanged=True,
                host_binding='torch C++ extension, Driver VMM/user objects, one retirement thread per owner',
                host_cflags=['-O2', '-std=c++17', '-pthread'],
                host_ldflags=['-lcuda', '-pthread', '-Wl,-z,nodelete'],
                target_qualified=False, host_syntax_checked=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EVIDENCE/'generated')
    args=parser.parse_args()
    original=BASE.read_bytes().decode()
    source=build(original)
    manifest=validate(source, original)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'candidate.py').write_bytes(source.encode())
    (args.out/'host.cpp').write_bytes(HOST.encode())
    (args.out/'generation.json').write_bytes((json.dumps(manifest, indent=2)+'\n').encode())
    print(json.dumps(manifest, indent=2))


if __name__=='__main__':
    main()
