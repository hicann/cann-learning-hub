# Precision Backend

The optimized backend uses BF16 storage for CSR matrix values and FP32 accumulation during SpMV.

This keeps GMRES residual checks in FP32 while reducing matrix memory footprint and SpMV traffic.
