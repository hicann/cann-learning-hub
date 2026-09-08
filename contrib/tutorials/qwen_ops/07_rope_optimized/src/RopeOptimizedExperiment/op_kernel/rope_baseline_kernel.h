#ifndef ROPE_BASELINE_KERNEL_H
#define ROPE_BASELINE_KERNEL_H

#include "kernel_operator.h"

using namespace AscendC;

#pragma pack(push, 1)
struct RoPeTiling {
    uint32_t totalTokens = 0;
    uint32_t headDim     = 0;
    uint32_t coreNum     = 1;
    uint32_t rowsPerCore = 0;
    uint32_t seqLen      = 0;
    uint32_t numHeads    = 1;
    uint32_t trigTokens  = 0;
    uint32_t compactTrig = 0;
    uint32_t tileSize    = 0;
};
#pragma pack(pop)

static_assert(sizeof(RoPeTiling) == 36, "Unexpected RoPe tiling data size");


class KernelRoPeBaseline {
public:
    __aicore__ inline KernelRoPeBaseline() {}

    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR cos_in, GM_ADDR sin_in, GM_ADDR output,
        uint32_t totalTokens, uint32_t headDim, uint32_t coreNum,
        uint32_t rowsPerCore, uint32_t seqLen, uint32_t numHeads,
        uint32_t trigTokens, uint32_t compactTrig, uint32_t tileSize)
    {
        totalTokens_ = totalTokens; headDim_ = headDim;
        coreNum_     = coreNum;     rowsPerCore_ = rowsPerCore;
        seqLen_      = seqLen;      numHeads_    = numHeads;
        trigTokens_  = trigTokens;  compactTrig_ = compactTrig;
        tileSize_    = tileSize;
        totalElements_ = totalTokens * headDim;
        trigElements_  = trigTokens * headDim;

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(input), totalElements_);
        cosGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(cos_in), trigElements_);
        sinGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(sin_in), trigElements_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(output), totalElements_);

        if (tileSize_ > 1) {
            uint32_t tileBytes = tileSize_ * headDim_ * sizeof(float);
            pipe_.InitBuffer(inQue_,  2, tileBytes);
            pipe_.InitBuffer(outQue_, 2, tileBytes);
            pipe_.InitBuffer(tmpBuf_, tileBytes);
            // compact cos/sin expansion buffer (only needed for compactTrig)
            if (compactTrig_) {
                pipe_.InitBuffer(ctBuf_, tileBytes);  // expanded cos tile
                pipe_.InitBuffer(stBuf_, tileBytes);  // expanded sin tile
            } else {
                pipe_.InitBuffer(cQue_,   2, tileBytes);
                pipe_.InitBuffer(sQue_,   2, tileBytes);
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        if (coreId >= coreNum_) return;
        uint32_t hh = headDim_ >> 1;
        uint32_t sr = coreId * rowsPerCore_;
        uint32_t er = (sr + rowsPerCore_ < totalTokens_)
                    ? (sr + rowsPerCore_) : totalTokens_;

        if (tileSize_ <= 1)
            processScalar(sr, er, hh);
        else
            processTiled(sr, er, hh);
    }

protected:
    __aicore__ inline void processScalar(uint32_t sr, uint32_t er, uint32_t hh)
    {
        uint32_t rpb=0,b=0,rib=0,sp=0;
        if(compactTrig_){rpb=numHeads_*seqLen_;b=sr/rpb;rib=sr-b*rpb;sp=rib%seqLen_;}
        for(uint32_t r=sr;r<er;r++){
            uint32_t ro=r*headDim_,tro=ro;
            if(compactTrig_)tro=(b*seqLen_+sp)*headDim_;
            for(uint32_t i=0;i<hh;i++){
                uint32_t i0=ro+i,i1=ro+i+hh,ti0=tro+i,ti1=tro+i+hh;
                float x0=inputGm_.GetValue(i0),x1=inputGm_.GetValue(i1);
                float c0=cosGm_.GetValue(ti0),s0=sinGm_.GetValue(ti0);
                float c1=cosGm_.GetValue(ti1),s1=sinGm_.GetValue(ti1);
                outputGm_.SetValue(i0,x0*c0-x1*s0);
                outputGm_.SetValue(i1,x1*c1+x0*s1);
            }
            if(compactTrig_){rib++;sp++;if(sp>=seqLen_)sp=0;if(rib>=rpb){rib=0;b++;sp=0;}}
        }
    }

    __aicore__ inline void processTiled(uint32_t sr, uint32_t er, uint32_t hh)
    {
        uint32_t rpb=0,b=0,rib=0,sp=0;
        if(compactTrig_){rpb=numHeads_*seqLen_;b=sr/rpb;rib=sr-b*rpb;sp=rib%seqLen_;}

        for(uint32_t ts=sr;ts<er;ts+=tileSize_){
            uint32_t te=(ts+tileSize_<er)?(ts+tileSize_):er;
            uint32_t cr=te-ts;

            // ★ CopyIn: DataCopy x → UB
            LocalTensor<float> xT=inQue_.AllocTensor<float>();
            DataCopy(xT,inputGm_[ts*headDim_],cr*headDim_);
            inQue_.EnQue(xT);

            if(compactTrig_==0){
                // Expand mode: cos/sin are 1:1 with x rows → direct DataCopy
                LocalTensor<float> cT=cQue_.AllocTensor<float>();
                LocalTensor<float> sT=sQue_.AllocTensor<float>();
                DataCopy(cT,cosGm_[ts*headDim_],cr*headDim_);
                DataCopy(sT,sinGm_[ts*headDim_],cr*headDim_);
                cQue_.EnQue(cT); sQue_.EnQue(sT);
            }else{
                // Compact mode: expand cos/sin into VECCALC buffers (not VECIN)
                LocalTensor<float> cE=ctBuf_.Get<float>();
                LocalTensor<float> sE=stBuf_.Get<float>();
                for(uint32_t r=0;r<cr;r++){
                    uint32_t base=(b*seqLen_+sp)*headDim_, off=r*headDim_;
                    for(uint32_t d=0;d<headDim_;d++){
                        cE.SetValue(off+d,cosGm_.GetValue(base+d));
                        sE.SetValue(off+d,sinGm_.GetValue(base+d));
                    }
                    rib++;sp++;if(sp>=seqLen_)sp=0;if(rib>=rpb){rib=0;b++;sp=0;}
                }

                // ★ Compute: Vector on UB
                LocalTensor<float> x=inQue_.DeQue<float>();
                LocalTensor<float> o=outQue_.AllocTensor<float>();
                LocalTensor<float> t=tmpBuf_.Get<float>();

                for(uint32_t r=0;r<cr;r++){
                    uint32_t ro=r*headDim_;
                    Mul(o[ro],    x[ro],    cE[ro],    static_cast<int32_t>(hh));
                    Mul(t[ro],    x[ro+hh], sE[ro],    static_cast<int32_t>(hh));
                    Sub(o[ro],    o[ro],    t[ro],    static_cast<int32_t>(hh));
                    Mul(t[ro],    x[ro],    sE[ro+hh], static_cast<int32_t>(hh));
                    Mul(o[ro+hh], x[ro+hh], cE[ro+hh], static_cast<int32_t>(hh));
                    Add(o[ro+hh], o[ro+hh], t[ro],    static_cast<int32_t>(hh));
                }
                outQue_.EnQue(o);
                inQue_.FreeTensor(x);

                LocalTensor<float> out=outQue_.DeQue<float>();
                DataCopy(outputGm_[ts*headDim_],out,cr*headDim_);
                outQue_.FreeTensor(out);
                continue;  // skip expand-mode code below
            }

            // ★ Compute: Vector on UB (expand-mode path)
            LocalTensor<float> x=inQue_.DeQue<float>();
            LocalTensor<float> c=cQue_.DeQue<float>();
            LocalTensor<float> s=sQue_.DeQue<float>();
            LocalTensor<float> o=outQue_.AllocTensor<float>();
            LocalTensor<float> t=tmpBuf_.Get<float>();

            for(uint32_t r=0;r<cr;r++){
                uint32_t ro=r*headDim_;
                Mul(o[ro],    x[ro],    c[ro],    static_cast<int32_t>(hh));
                Mul(t[ro],    x[ro+hh], s[ro],    static_cast<int32_t>(hh));
                Sub(o[ro],    o[ro],    t[ro],    static_cast<int32_t>(hh));
                Mul(t[ro],    x[ro],    s[ro+hh], static_cast<int32_t>(hh));
                Mul(o[ro+hh], x[ro+hh], c[ro+hh], static_cast<int32_t>(hh));
                Add(o[ro+hh], o[ro+hh], t[ro],    static_cast<int32_t>(hh));
            }
            outQue_.EnQue(o);
            inQue_.FreeTensor(x); cQue_.FreeTensor(c); sQue_.FreeTensor(s);

            LocalTensor<float> out=outQue_.DeQue<float>();
            DataCopy(outputGm_[ts*headDim_],out,cr*headDim_);
            outQue_.FreeTensor(out);
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN,2>  inQue_;
    TQue<TPosition::VECIN,2>  cQue_;
    TQue<TPosition::VECIN,2>  sQue_;
    TQue<TPosition::VECOUT,2> outQue_;
    TBuf<TPosition::VECCALC>   tmpBuf_;
    TBuf<TPosition::VECCALC>   ctBuf_, stBuf_;  // compact cos/sin expansion

    GlobalTensor<float> inputGm_,cosGm_,sinGm_,outputGm_;
    uint32_t totalTokens_,headDim_,coreNum_,rowsPerCore_,totalElements_;
    uint32_t seqLen_,numHeads_,trigTokens_,trigElements_,compactTrig_;
    uint32_t tileSize_;
};

#endif
