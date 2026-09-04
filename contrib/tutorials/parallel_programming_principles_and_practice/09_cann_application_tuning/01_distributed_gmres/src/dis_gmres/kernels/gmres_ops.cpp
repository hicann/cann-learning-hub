#include "kernel_operator.h"

extern "C" __global__ __vector__ __aicore__ void gmres_spmv(GM_ADDR row, GM_ADDR col, GM_ADDR val,
                                                   GM_ADDR x, GM_ADDR y, int64_t rows) {
  // Contiguous 16-float (64 B cacheline) aligned partition: no two cores
  // ever write the same 64 B cacheline of the float output.
  constexpr int64_t kFloatsPerCacheLine = 16;
  const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
  const int64_t elementsPerBlock = ((rows + blockCount - 1) / blockCount + kFloatsPerCacheLine - 1) / kFloatsPerCacheLine * kFloatsPerCacheLine;
  const int64_t first = static_cast<int64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
  const int64_t last = first + elementsPerBlock < rows ? first + elementsPerBlock : rows;
  AscendC::GlobalTensor<int32_t> r, c; AscendC::GlobalTensor<float> a, vx, vy;
  r.SetGlobalBuffer((__gm__ int32_t*)row); c.SetGlobalBuffer((__gm__ int32_t*)col);
  a.SetGlobalBuffer((__gm__ float*)val); vx.SetGlobalBuffer((__gm__ float*)x);
  vy.SetGlobalBuffer((__gm__ float*)y);
  for (int64_t i=first;i<last;++i){float s=0;for(int32_t p=r.GetValue((int32_t)i);p<r.GetValue((int32_t)(i+1));++p)s+=a.GetValue(p)*vx.GetValue(c.GetValue(p));vy.SetValue((int32_t)i,s);}
}
extern "C" __global__ __vector__ __aicore__ void gmres_dot(GM_ADDR x, GM_ADDR y, GM_ADDR out,
                                                  int64_t n) {
  // One AI Core deliberately owns the reduction. This is a correctness baseline;
  // profiler-guided tiled reduction is the tuning exercise.
  if (AscendC::GetBlockIdx()!=0) return; AscendC::GlobalTensor<float> a,b,o;
  a.SetGlobalBuffer((__gm__ float*)x);b.SetGlobalBuffer((__gm__ float*)y);o.SetGlobalBuffer((__gm__ float*)out);
  float s=0;for(int64_t i=0;i<n;++i)s+=a.GetValue((int32_t)i)*b.GetValue((int32_t)i);o.SetValue(0,s);
}
extern "C" __global__ __vector__ __aicore__ void gmres_axpy(GM_ADDR x, GM_ADDR y, int64_t n,
                                                   float alpha) {
  // Contiguous 16-float (64 B cacheline) aligned partition: no two cores
  // ever write the same 64 B cacheline of the float output.
  constexpr int64_t kFloatsPerCacheLine = 16;
  const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
  const int64_t elementsPerBlock = ((n + blockCount - 1) / blockCount + kFloatsPerCacheLine - 1) / kFloatsPerCacheLine * kFloatsPerCacheLine;
  const int64_t first = static_cast<int64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
  const int64_t last = first + elementsPerBlock < n ? first + elementsPerBlock : n;
  AscendC::GlobalTensor<float>a,b;a.SetGlobalBuffer((__gm__ float*)x);b.SetGlobalBuffer((__gm__ float*)y);
  for(int64_t i=first;i<last;++i)b.SetValue((int32_t)i,b.GetValue((int32_t)i)+alpha*a.GetValue((int32_t)i));
}
extern "C" __global__ __vector__ __aicore__ void gmres_scale(GM_ADDR x, int64_t n, float alpha) {
  // Contiguous 16-float (64 B cacheline) aligned partition: no two cores
  // ever write the same 64 B cacheline of the float output.
  constexpr int64_t kFloatsPerCacheLine = 16;
  const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
  const int64_t elementsPerBlock = ((n + blockCount - 1) / blockCount + kFloatsPerCacheLine - 1) / kFloatsPerCacheLine * kFloatsPerCacheLine;
  const int64_t first = static_cast<int64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
  const int64_t last = first + elementsPerBlock < n ? first + elementsPerBlock : n;
  AscendC::GlobalTensor<float>a;a.SetGlobalBuffer((__gm__ float*)x);
  for(int64_t i=first;i<last;++i)a.SetValue((int32_t)i,alpha*a.GetValue((int32_t)i));
}
extern "C" __global__ __vector__ __aicore__ void gmres_sub(GM_ADDR a0, GM_ADDR b0, GM_ADDR out,
                                                  int64_t n) {
  // Contiguous 16-float (64 B cacheline) aligned partition: no two cores
  // ever write the same 64 B cacheline of the float output.
  constexpr int64_t kFloatsPerCacheLine = 16;
  const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
  const int64_t elementsPerBlock = ((n + blockCount - 1) / blockCount + kFloatsPerCacheLine - 1) / kFloatsPerCacheLine * kFloatsPerCacheLine;
  const int64_t first = static_cast<int64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
  const int64_t last = first + elementsPerBlock < n ? first + elementsPerBlock : n;
  AscendC::GlobalTensor<float>a,b,o;a.SetGlobalBuffer((__gm__ float*)a0);b.SetGlobalBuffer((__gm__ float*)b0);o.SetGlobalBuffer((__gm__ float*)out);
  for(int64_t i=first;i<last;++i)o.SetValue((int32_t)i,a.GetValue((int32_t)i)-b.GetValue((int32_t)i));
}
