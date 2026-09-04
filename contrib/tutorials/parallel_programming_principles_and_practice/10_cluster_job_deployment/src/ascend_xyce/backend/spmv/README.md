# SpMV Backend

The SpMV implementation is reused from Ascend-GMRES:

- CSR matrix
- BF16 values
- FP32 accumulation
- persistent CSR resident design

The Xyce adapter calls the backend through the GMRES linear solver interface.
