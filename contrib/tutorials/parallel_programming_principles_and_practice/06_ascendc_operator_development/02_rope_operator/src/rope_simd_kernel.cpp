// RoPE SIMD Kernel：Ascend 910B3 Vector Core，RTC 目标 dav-2201。
// Host 已把 [12,64] 输入转换为 384 个连续 even/odd pair；Kernel 使用
// DataCopy 和 Mul/Sub/Add 批量计算，不在设备侧做逐元素标量循环。
// out_even = x_even * cos - x_odd * sin
// out_odd  = x_even * sin + x_odd * cos

#include "kernel_operator.h"

constexpr int32_t TOTAL_PAIRS = 384;              // 12 heads * 32 pairs = 384 连续 pair
constexpr int32_t BUFFER_NUM  = 1;                // KISS：单 tile、单缓冲，一次搬入全部数据

class KernelRopeSimd {
public:
    __aicore__ inline KernelRopeSimd() {}

    __aicore__ inline void Init(GM_ADDR xEven, GM_ADDR xOdd, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR outEven, GM_ADDR outOdd)
    {
        xEvenGm.SetGlobalBuffer((__gm__ float*)xEven, TOTAL_PAIRS);
        xOddGm.SetGlobalBuffer((__gm__ float*)xOdd, TOTAL_PAIRS);
        cosGm.SetGlobalBuffer((__gm__ float*)cos, TOTAL_PAIRS);
        sinGm.SetGlobalBuffer((__gm__ float*)sin, TOTAL_PAIRS);
        outEvenGm.SetGlobalBuffer((__gm__ float*)outEven, TOTAL_PAIRS);
        outOddGm.SetGlobalBuffer((__gm__ float*)outOdd, TOTAL_PAIRS);

        pipe.InitBuffer(inQueueXEven, BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(inQueueXOdd,  BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(inQueueCos,   BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(inQueueSin,   BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(tmpEven1,     BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(tmpEven2,     BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(tmpOdd1,      BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(tmpOdd2,      BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(outQueueEven, BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
        pipe.InitBuffer(outQueueOdd,  BUFFER_NUM, TOTAL_PAIRS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> xe = inQueueXEven.AllocTensor<float>();
        AscendC::LocalTensor<float> xo = inQueueXOdd.AllocTensor<float>();
        AscendC::LocalTensor<float> c  = inQueueCos.AllocTensor<float>();
        AscendC::LocalTensor<float> s  = inQueueSin.AllocTensor<float>();
        AscendC::DataCopy(xe, xEvenGm, TOTAL_PAIRS);
        AscendC::DataCopy(xo, xOddGm, TOTAL_PAIRS);
        AscendC::DataCopy(c,  cosGm,  TOTAL_PAIRS);
        AscendC::DataCopy(s,  sinGm,  TOTAL_PAIRS);
        inQueueXEven.EnQue(xe);
        inQueueXOdd.EnQue(xo);
        inQueueCos.EnQue(c);
        inQueueSin.EnQue(s);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xe = inQueueXEven.DeQue<float>();
        AscendC::LocalTensor<float> xo = inQueueXOdd.DeQue<float>();
        AscendC::LocalTensor<float> c  = inQueueCos.DeQue<float>();
        AscendC::LocalTensor<float> s  = inQueueSin.DeQue<float>();

        AscendC::LocalTensor<float> te1 = tmpEven1.AllocTensor<float>();
        AscendC::LocalTensor<float> te2 = tmpEven2.AllocTensor<float>();
        AscendC::LocalTensor<float> to1 = tmpOdd1.AllocTensor<float>();
        AscendC::LocalTensor<float> to2 = tmpOdd2.AllocTensor<float>();
        AscendC::LocalTensor<float> oe  = outQueueEven.AllocTensor<float>();
        AscendC::LocalTensor<float> oo  = outQueueOdd.AllocTensor<float>();

        // out_even = x_even * cos - x_odd * sin
        AscendC::Mul(te1, xe, c, TOTAL_PAIRS);
        AscendC::Mul(te2, xo, s, TOTAL_PAIRS);
        AscendC::Sub(oe, te1, te2, TOTAL_PAIRS);
        // out_odd = x_even * sin + x_odd * cos
        AscendC::Mul(to1, xe, s, TOTAL_PAIRS);
        AscendC::Mul(to2, xo, c, TOTAL_PAIRS);
        AscendC::Add(oo, to1, to2, TOTAL_PAIRS);

        outQueueEven.EnQue(oe);
        outQueueOdd.EnQue(oo);

        tmpEven1.FreeTensor(te1);
        tmpEven2.FreeTensor(te2);
        tmpOdd1.FreeTensor(to1);
        tmpOdd2.FreeTensor(to2);
        inQueueXEven.FreeTensor(xe);
        inQueueXOdd.FreeTensor(xo);
        inQueueCos.FreeTensor(c);
        inQueueSin.FreeTensor(s);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> oe = outQueueEven.DeQue<float>();
        AscendC::LocalTensor<float> oo = outQueueOdd.DeQue<float>();
        AscendC::DataCopy(outEvenGm, oe, TOTAL_PAIRS);
        AscendC::DataCopy(outOddGm, oo, TOTAL_PAIRS);
        outQueueEven.FreeTensor(oe);
        outQueueOdd.FreeTensor(oo);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>  inQueueXEven, inQueueXOdd, inQueueCos, inQueueSin;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueEven, outQueueOdd;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpEven1, tmpEven2, tmpOdd1, tmpOdd2;
    AscendC::GlobalTensor<float> xEvenGm, xOddGm, cosGm, sinGm, outEvenGm, outOddGm;
};

extern "C" __global__ __aicore__ void rope_simd(GM_ADDR xEven, GM_ADDR xOdd, GM_ADDR cos,
                                                GM_ADDR sin, GM_ADDR outEven, GM_ADDR outOdd)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelRopeSimd op;
    op.Init(xEven, xOdd, cos, sin, outEven, outOdd);
    op.Process();
}
