"""Student exercise: fill only the tile-start expression; answer is provided separately."""
import tilelang
import tilelang.language as T
import torch,torch_npu
N,TILE,NUM_BLOCKS=1179648,1024,8
@T.prim_func
def main(A:T.Tensor((N,),'float32'),Y:T.Tensor((N,),'float32')):
    with T.Kernel(NUM_BLOCKS) as bx:
        a=T.alloc_shared((TILE,),'float32')
        y=T.alloc_shared((TILE,),'float32')
        for it in T.serial(N//(NUM_BLOCKS*TILE)):
            begin=0  # TODO: only replace this index expression.
            T.copy(A[begin:begin+TILE],a)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for group in T.serial(TILE // 64):
                    ra = T.simd.vld(a[group * 64])
                    ry = T.simd.vexp(ra, mask)
                    T.simd.vsts(y[group * 64], ry, mask)
            T.copy(y,Y[begin:begin+TILE])
if __name__=='__main__':
    from pathlib import Path
    if any(line.strip().startswith("begin=0") and "TODO" in line
           for line in Path(__file__).read_text().splitlines()):
        raise SystemExit("待完成练习：请先补全 begin 的分核与分块索引；完整答案见 answer/03.03_tiled_exp.py。当前不启动 NPU。")
    k=tilelang.compile(main,out_idx=-1)
    x=torch.linspace(-4,4,N,device='npu')
    y=k(x);torch.npu.synchronize()
    torch.testing.assert_close(y,torch.exp(x),rtol=1e-5,atol=1e-6)
    print('Verification passed!')
