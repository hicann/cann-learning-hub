# Xyce Source Directory

`scripts/fetch_xyce.sh` clones the upstream Xyce source tree into `third_party/Xyce/source`.

The Ascend-Xyce benchmark does not modify Xyce core source files. It uses a wrapper/adapter layer that mirrors the Xyce linear solver path and calls the Ascend-GMRES backend through a `solve(A,b,x)` interface.
