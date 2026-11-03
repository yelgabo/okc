import ctypes,json
entries=[('/usr/lib/wsl/lib/libcuda.so',['cuGraphGetNodes','cuGraphNodeGetType','cuGraphKernelNodeGetParams_v2','cuGraphGetEdges_v2']),('/root/.cache/okc-cuda130/nvidia/cu13/lib/libcudart.so.13',['cudaGetFuncBySymbol'])]
result=[]
for path,names in entries:
 try:
  library=ctypes.CDLL(path)
  resolved={}
  for name in names:
   try: resolved[name]=bool(getattr(library,name))
   except AttributeError: resolved[name]=False
  result.append(dict(path=path,symbols=resolved))
 except OSError as error: result.append(dict(path=path,error=str(error)))
print(json.dumps(dict(scope='Local library load and symbol resolution only; no CUDA API invoked, no host compilation or target proof',libraries=result),indent=2))
