
import linecache
import types
import torch
import json

def _pair_module(name, source):
    module = types.ModuleType(name)
    filename = "<" + name + ">"
    module.__file__ = filename
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), module.__dict__)
    return module

source = 'import triton\nimport triton.language as tl\nBIAS=tl.constexpr(3)\n@triton.jit\ndef probe(X):\n    tl.store(X, tl.load(X) + BIAS)\n'
module = _pair_module('_pair_loader_cuda_probe', source)
x=torch.zeros(1,device='cuda')
module.probe[(1,)](x)
torch.cuda.synchronize()
assert x.item()==3
print('DYNAMIC_TRITON_MODULE_PASS',torch.cuda.get_device_name())
