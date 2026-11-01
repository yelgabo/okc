"""Build an off-board CUPTI availability check around the official layer baseline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CPP = r'''
#include <cupti.h>
#include <cupti_profiler_target.h>
#include <sstream>
#include <vector>

std::string counter_preflight() {
    std::ostringstream report;
    auto record = [&](const char* step, CUptiResult status) {
        const char* message = nullptr;
        cuptiGetResultString(status, &message);
        report << step << "=" << int(status) << "(" << (message ? message : "?") << ") ";
        return status == CUPTI_SUCCESS;
    };
    CUpti_Profiler_Initialize_Params init{CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    if (!record("initialize", cuptiProfilerInitialize(&init))) return report.str();
    CUpti_Profiler_GetCounterAvailability_Params query{
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    if (record("availability_size", cuptiProfilerGetCounterAvailability(&query))) {
        report << "bytes=" << query.counterAvailabilityImageSize << " ";
        if (query.counterAvailabilityImageSize && query.counterAvailabilityImageSize <= (16u << 20)) {
            std::vector<uint8_t> buffer(query.counterAvailabilityImageSize);
            query.pCounterAvailabilityImage = buffer.data();
            record("availability_image", cuptiProfilerGetCounterAvailability(&query));
        }
    }
    CUpti_Profiler_DeInitialize_Params fini{CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    record("deinitialize", cuptiProfilerDeInitialize(&fini));
    return report.str();
}
'''
WRAPPER = '''
_baseline_prepare = prepare

def prepare(weights):
    from torch.utils.cpp_extension import load_inline
    state = _baseline_prepare(weights)
    module = load_inline(
        name="layer_counter_availability_v1", cpp_sources=_COUNTER_SOURCE,
        functions=["counter_preflight"], with_cuda=True,
        extra_include_paths=["/usr/local/cuda/extras/CUPTI/include"],
        extra_ldflags=["-L/usr/local/cuda/extras/CUPTI/lib64", "-lcupti"])
    print("COUNTER_PREFLIGHT", module.counter_preflight(), flush=True)
    return state
'''

if __name__ == '__main__':
    baseline = (ROOT/'muse_decode_layer/baseline.py').read_text(encoding='utf-8')
    source = '# okc: problem=muse_decode_layer\n' + baseline
    source += '\n_COUNTER_SOURCE = ' + repr(CPP) + '\n' + WRAPPER
    compile(source, '<counter-preflight>', 'exec')
    Path(sys.argv[1]).write_text(source, encoding='utf-8', newline='\n')
