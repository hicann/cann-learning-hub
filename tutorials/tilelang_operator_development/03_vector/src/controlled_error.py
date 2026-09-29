"""Intentionally fails on the Host, before compiling or launching an oversized tile."""
from vector_kernels import unary_serial
unary_serial(65536,65536,1,'relu')
