# GMRES Backend

This directory is reserved for the Ascend-GMRES dependency.

`scripts/prepare_backend.sh` resolves the backend in this order:

1. `ASCEND_GMRES_DIR`
2. sibling directory `../Ascend-GMRES`
3. clone `git@gitcode.com:maeveyixue/Ascend-GMRES.git` into `backend/gmres/Ascend-GMRES`

The source is reused at build time instead of copied into this repository.
