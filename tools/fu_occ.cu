// Max co-resident clusters for a 416-thread, 1-CTA-per-SM kernel at the fused pooler's shared-memory size.
// Usage on the box: nvcc -arch=sm_90a -o fu_occ tools/fu_occ.cu && flock .build/box/gpu.lock ./fu_occ
#include <cstdio>
#include <cuda_runtime.h>
__global__ void __launch_bounds__(416, 1) k(int* p) { extern __shared__ char s[]; if (p) p[threadIdx.x] = s[threadIdx.x]; }
int main() {
  int optin = 0, nsm = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
  cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  printf("optin %d sms %d\n", optin, nsm);
  for (int cl : {2, 4, 8, 16})
    for (int sm : {100000, 180000, 206336, 214000, 220000, optin}) {
      cudaLaunchConfig_t c = {};
      c.gridDim = dim3(128);
      c.blockDim = dim3(416);
      c.dynamicSmemBytes = sm;
      cudaLaunchAttribute a[1];
      a[0].id = cudaLaunchAttributeClusterDimension;
      a[0].val.clusterDim.x = cl;
      a[0].val.clusterDim.y = 1;
      a[0].val.clusterDim.z = 1;
      c.attrs = a;
      c.numAttrs = 1;
      int n = -1;
      cudaError_t e = cudaOccupancyMaxActiveClusters(&n, k, &c);
      printf("cluster %2d smem %6d -> max active clusters %3d (%s), CTAs %d\n", cl, sm, n, cudaGetErrorString(e), n * cl);
    }
}
