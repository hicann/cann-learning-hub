# Xyce Linear Solver Flow

This project does not modify Xyce core source code. It models the Xyce linear solver path through a wrapper and connects that wrapper to Ascend-GMRES.

## Call Chain

```text
Xyce application
  |
  | assemble nonlinear/device equations
  v
Linear system
  |
  | sparse matrix A, RHS b, solution x
  v
Linear solver interface
  |
  | solve(A, b, x)
  v
GMRES
  |
  | SpMV, dot, axpy, norm
  v
Device-resident GMRES backend
  |
  | FP32 CSR, SpMV/Dot/Norm/AXPY/Scale Ascend C kernels
  v
Ascend AI Core
```

## Xyce Concepts Mapped by the Wrapper

- Matrix assembly: represented by `XyceApplicationWrapper`, which builds `b = A*x_true`.
- Sparse matrix representation: represented by `XyceSparseMatrixAdapter`, using CSR `row_ptr`, `col_idx`, `values`.
- Linear solve interface: represented by `XyceLinearSolverAdapter::solve(A,b,x)`.
- Device GMRES backend: reused from the chapter 07 `dis_gmres` runtime.
- Matrix and Krylov vectors remain in Device memory for the solve; Host keeps only Hessenberg/Givens scalars.

## Solver Replacement

The wrapper exposes three solver modes:

- `Xyce CPU single GMRES`
- `Xyce CPU OpenMP16 GMRES`
- `Xyce Ascend Device GMRES (Ascend C RTC)`

The Ascend path preserves:

- tolerance: `1e-6`
- max iterations: `10000`
- restart: `30`
- FP32 CSR uploaded once during solver initialization
- persistent DeviceVector design so full vectors are not copied through Host inside Arnoldi

## Benchmark Path

```text
benchmark/xyce_benchmark.cpp
  |
  v
XyceApplicationWrapper
  |
  v
XyceLinearSolverAdapter
  |
  v
Ascend-GMRES solver
  |
  v
SpMV / BLAS-1 profiling
```
