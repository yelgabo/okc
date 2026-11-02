
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
#include <vector>

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
  bool retained=false, reserved=false, created=false, mapped=false;
  std::vector<CUevent> completions;
  std::atomic<bool> callback_finished{false};

  void cleanup(CUstreamCaptureMode* previous=nullptr) noexcept {
    auto check=[previous](CUresult result, const char* stage) noexcept {
      if (result != CUDA_SUCCESS && previous) {
        CUstreamCaptureMode restore=*previous;
        fatal(cuThreadExchangeStreamCaptureMode(&restore), "cleanup-mode-error-restore");
      }
      fatal(result, stage);
    };
    if (context) {
      check(cuCtxPushCurrent(context), "cleanup-context");
      for (CUevent event : completions) {
        check(cuEventSynchronize(event), "owner-completion");
        check(cuEventDestroy(event), "owner-event-destroy");
      }
      if (mapped) check(cuMemUnmap(address, bytes), "unmap");
      if (created) check(cuMemRelease(handle), "release");
      if (reserved) check(cuMemAddressFree(address, bytes), "address-free");
      CUcontext popped{};
      check(cuCtxPopCurrent(&popped), "cleanup-pop");
      if (popped != context) check(CUDA_ERROR_INVALID_CONTEXT, "cleanup-pop-identity");
    }
    if (retained) check(cuDevicePrimaryCtxRelease(device), "primary-release");
  }
};

static void CUDA_CB graph_release(void* opaque) noexcept {
  static_cast<Mapping*>(opaque)->callback_finished.store(true, std::memory_order_release);
}

static void retirement(Mapping* mapping) noexcept {
  while (!mapping->callback_finished.load(std::memory_order_acquire))
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CUstreamCaptureMode previous=CU_STREAM_CAPTURE_MODE_RELAXED;
  fatal(cuThreadExchangeStreamCaptureMode(&previous), "cleanup-mode-enter");
  mapping->cleanup(&previous);
  fatal(cuThreadExchangeStreamCaptureMode(&previous), "cleanup-mode-restore");
  delete mapping;
}

struct Allocation {
  torch::Tensor tensor;
  Mapping* mapping;
  Allocation(torch::Tensor value, Mapping* owner): tensor(std::move(value)), mapping(owner) {}

  uint64_t before_read(uint64_t stream_bits) {
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
      return 0;
    }
    CUevent event{};
    require(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "reader-event-create");
    try { mapping->completions.push_back(event); }
    catch (...) { fatal(cuEventDestroy(event), "unsubmitted-event-destroy"); throw; }
    return reinterpret_cast<uint64_t>(event);
  }

  void after_read(uint64_t stream_bits, uint64_t event_bits) {
    if (event_bits) fatal(cuEventRecord(reinterpret_cast<CUevent>(event_bits),
                          reinterpret_cast<CUstream>(stream_bits)), "reader-event-record");
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
    CUevent copy_event{};
    require(cuEventCreate(&copy_event, CU_EVENT_DISABLE_TIMING), "copy-event-create");
    try { owner->completions.push_back(copy_event); }
    catch (...) { fatal(cuEventDestroy(copy_event), "unsubmitted-copy-event"); throw; }
    CUresult copy_result=cuMemcpyDtoDAsync(owner->address,
            reinterpret_cast<CUdeviceptr>(source.data_ptr()), logical, stream);
    fatal(cuEventRecord(copy_event, stream), "copy-event-record");
    fatal(cuEventSynchronize(copy_event), "copy-completion");
    require(copy_result, "bit-copy");
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
    .def("before_read", &Allocation::before_read)
    .def("after_read", &Allocation::after_read);
  module.def("allocate", &allocate);
}
