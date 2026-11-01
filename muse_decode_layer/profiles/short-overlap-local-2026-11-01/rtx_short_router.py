"""Local GPU correctness fixture. H200-compiled pooler and whole-layer timings are excluded."""
import ast
import ctypes as C
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "muse_decode_layer/cand/g1b4_stage_release.py"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"
tree = ast.parse(SOURCE.read_text(encoding="utf8"))
router = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and getattr(node.targets[0], "id", "") == "_ROUTER_FP32_CUDA")
device = router[router.index("__global__"):router.index("int router_fp32_launch(")]
assert "router_fp32_kernel<<<256, 256, 0, (cudaStream_t)stream>>>" in router
ptr, string, uint = C.c_void_p, C.c_char_p, C.c_uint
def call(lib, name, types, *args):
    fn = getattr(lib, name)
    fn.argtypes = types
    rc = fn(*args)
    if rc: raise RuntimeError((name, rc))

def compile_ptx():
    lib = C.CDLL("/root/.cache/okc-cuda128/packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12")
    prog = ptr()
    call(lib, "nvrtcCreateProgram", [C.POINTER(ptr),string,string,C.c_int,ptr,ptr],
         C.byref(prog), ('extern "C" '+device).encode(), b"router.cu", 0, None, None)
    opts = (string*1)(b"--gpu-architecture=compute_120")
    call(lib, "nvrtcCompileProgram", [ptr,C.c_int,C.POINTER(string)], prog, 1, opts)
    size = C.c_size_t()
    call(lib,"nvrtcGetPTXSize",[ptr,C.POINTER(C.c_size_t)],prog,C.byref(size))
    out = C.create_string_buffer(size.value)
    call(lib,"nvrtcGetPTX",[ptr,ptr],prog,out)
    call(lib,"nvrtcDestroyProgram",[C.POINTER(ptr)],C.byref(prog))
    return out.value

class Router:
    def __init__(self):
        import torch
        torch.cuda.init()
        torch.empty(1, device="cuda")
        self.lib = C.CDLL("libcuda.so.1")
        self.module, self.function = ptr(), ptr()
        call(self.lib,"cuModuleLoadData",[C.POINTER(ptr),string],C.byref(self.module),compile_ptx())
        call(self.lib,"cuModuleGetFunction",[C.POINTER(ptr),ptr,string],C.byref(self.function),self.module,b"router_fp32_kernel")

    def router_fp32_launch(self,x,w,out,m,stream):
        assert 1 <= m <= 8
        values = [ptr(x),ptr(w),ptr(out),C.c_int(m)]
        args = (ptr*4)(*(C.addressof(v) for v in values))
        call(self.lib,"cuLaunchKernel",[ptr]+[uint]*7+[ptr,C.POINTER(ptr),ptr],
             self.function,256,1,1,256,1,1,0,ptr(stream),args,None)
        return 0

if __name__ == "__main__":
    import torch
    torch.manual_seed(9187)
    ext = Router()
    print(torch.__version__,torch.cuda.get_device_name(),hashlib.sha256(device.encode()).hexdigest(),flush=True)
    w = (torch.randn(256,8192,device="cuda")/8192**0.5).contiguous()
    for m in (1,8):
        x = torch.randn(m,8192,device="cuda")
        out = torch.empty(m,256,device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ext.router_fp32_launch(x.data_ptr(),w.data_ptr(),out.data_ptr(),m,stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(stream)
        ref = (x.double()@w.double().T).float()
        torch.testing.assert_close(out,ref,rtol=1e-5,atol=1e-5)
        print("FP32_ROUTER_PASS",m,(out-ref).abs().max().item(),flush=True)
