
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <algorithm>
constexpr int JMAX=88,RB_A=512;
struct Params {int* doneA;};
struct Smem {int njS,njF;};
struct Dim {int x;};
Dim gridDim;
thread_local int lane;
std::barrier warp_barrier(32);
int ready_bits[32],vote;
std::atomic<int> polls,completed;
std::atomic<bool> touched[136];
int* storage;
int ld_acquire(int* ptr) {
  int c=ptr-storage-128;
  if(c<0||c>=gridDim.x) std::terminate();
  touched[c].store(true,std::memory_order_relaxed);
  polls.fetch_add(1,std::memory_order_relaxed);
  return std::atomic_ref<int>(*ptr).load(std::memory_order_acquire);
}
int __all_sync(unsigned mask,int value) {
  if(mask!=0xffffffffu)std::terminate();
  ready_bits[lane]=value;warp_barrier.arrive_and_wait();
  if(lane==0){vote=1;for(int x:ready_bits)vote&=x;}
  warp_barrier.arrive_and_wait();int result=vote;warp_barrier.arrive_and_wait();return result;
}
void __nanosleep(int) {std::this_thread::yield();}
#define __device__
#define __forceinline__ inline

static_assert(JMAX <= 128 && 128 + 136 <= 512, "short completion flags overlap scratch");
__device__ __forceinline__ void short_ready_wait(const Params& p,const Smem& S,int job,int lane) {
  const int total = S.njS * RB_A, G = gridDim.x;
  const int L = (job-S.njF) * RB_A, R = L + RB_A;
  const int first = ((L+1)*G + total-1)/total - 1;
  const int last = (R*G + total-1)/total - 1;
  while (true) {
    int ready = 1;
    for (int c=first+lane;c<=last;c+=32) ready &= ld_acquire(p.doneA+128+c) != 0;
    if (__all_sync(0xffffffffu,ready)) break;
    __nanosleep(64);
  }
}

int main() {
  const int cases[][3]={{1,1,0},{132,1,0},{132,2,0},{132,2,1},{132,10,0},
                       {132,10,4},{132,10,9},{136,32,0},{136,32,15},{136,32,31},{114,17,8},{3,32,5}};
  int tests=0;
  for(const auto& config:cases)for(int epoch=1;epoch<=2;++epoch){
    int flags[512]={};std::atomic<int> data[136]{};storage=flags;gridDim.x=config[0];
    int jobs=config[1],job=config[2],missing=-1;
    bool expected[136]={};
    for(int c=0;c<gridDim.x;++c){
      int a=jobs*512*c/gridDim.x,b=jobs*512*(c+1)/gridDim.x;
      expected[c]=std::max(a,job*512)<std::min(b,(job+1)*512);
      if(expected[c])missing=c;
      touched[c].store(false);
    }
    for(int c=0;c<gridDim.x;++c)if(expected[c]&&c!=missing){
      data[c].store(epoch,std::memory_order_relaxed);std::atomic_ref<int>(flags[128+c]).store(1,std::memory_order_release);
    }
    polls=0;completed=0;std::atomic<bool> bad{false};std::vector<std::thread> threads;
    for(int l=0;l<32;++l)threads.emplace_back([&,l]{
      lane=l;short_ready_wait(Params{flags},Smem{jobs,0},job,l);
      warp_barrier.arrive_and_wait();
      for(int c=0;c<gridDim.x;++c)if(expected[c]&&data[c].load(std::memory_order_relaxed)!=epoch)bad=true;
      completed.fetch_add(1);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    bool premature=completed.load()!=0;
    data[missing].store(epoch,std::memory_order_relaxed);
    std::atomic_ref<int>(flags[128+missing]).store(1,std::memory_order_release);
    for(auto& t:threads)t.join();
    if(premature||bad.load()){std::puts("premature completion or missing visibility");return 1;}
    for(int c=0;c<gridDim.x;++c)if(touched[c].load()!=expected[c]){
      std::puts("incorrect contributor set");return 2;
    }
    std::fill(flags,flags+512,0);++tests;
  }
  std::printf("%d native wait cases, 32 lanes, withheld contributor and visibility passed\n",tests);
}
