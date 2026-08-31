#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy test for the custom AsStrided operator.

The NPU result is produced by the generated custom aclnn API:
    aclnnAsStridedGetWorkspaceSize -> aclnnAsStrided

Before running:
    1. Build/package/install the custom operator.
    2. Source the installed custom OPP set_env.bash.
    3. Optionally set AS_STRIDED_OPAPI_LIB to the exact libcust_opapi.so.
"""

import ctypes
import os
from pathlib import Path
from typing import Sequence

import numpy as np


ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_FORMAT_ND = 2

ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT32 = 3

DTYPE_TO_ACL = {
    np.dtype(np.float32): ACL_FLOAT,
    np.dtype(np.float16): ACL_FLOAT16,
    np.dtype(np.int32): ACL_INT32,
}


def check_acl(ret: int, api: str) -> None:
    if ret != ACL_SUCCESS:
        hint = ""
        if ret == 561003:
            hint = (
                " (kernel not found; make sure the custom OPP is installed "
                "and its set_env.bash has been sourced)"
            )
        raise RuntimeError(f"{api} failed, ACL error code: {ret}{hint}")


def contiguous_strides(shape: Sequence[int]) -> list[int]:
    if not shape:
        return []

    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * int(shape[i + 1])
    return strides


def find_opapi_lib() -> str:
    env_path = os.environ.get("AS_STRIDED_OPAPI_LIB")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"AS_STRIDED_OPAPI_LIB does not exist: {path}"
            )
        return str(path)

    script_dir = Path(__file__).resolve().parent
    local_build = script_dir / "code" / "build" / "op_host" / "libcust_opapi.so"

    if local_build.is_file():
        return str(local_build)

    raise FileNotFoundError(
        "Cannot find AsStrided libcust_opapi.so. "
        "Build the operator first or set AS_STRIDED_OPAPI_LIB."
    )


def as_strided_golden(
    input_x: np.ndarray,
    size: Sequence[int],
    stride: Sequence[int],
    storage_offset: int,
) -> np.ndarray:
    input_x = np.ascontiguousarray(input_x)
    size = tuple(int(v) for v in size)
    stride = tuple(int(v) for v in stride)

    if len(size) != len(stride):
        raise ValueError("size and stride must have the same rank")

    storage = input_x.reshape(-1)
    output = np.empty(size, dtype=input_x.dtype)

    for out_index in np.ndindex(size):
        src_index = int(storage_offset)
        for index, step in zip(out_index, stride):
            src_index += int(index) * int(step)

        if not 0 <= src_index < storage.size:
            raise ValueError(
                f"AsStrided access out of range: "
                f"output_index={out_index}, storage_index={src_index}"
            )

        output[out_index] = storage[src_index]

    return output


class AclnnAsStrided:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.stream = ctypes.c_void_p()
        self.initialized = False
        self.device_set = False
        self.stream_created = False

        mode = getattr(ctypes, "RTLD_GLOBAL", 0)

        self.acl = ctypes.CDLL("libascendcl.so", mode=mode)
        self.nnopbase = ctypes.CDLL("libnnopbase.so", mode=mode)

        opapi_path = find_opapi_lib()
        self.opapi = ctypes.CDLL(opapi_path, mode=mode)

        self._bind()

        if not hasattr(self.opapi, "aclnnAsStridedGetWorkspaceSize"):
            raise RuntimeError(
                f"{opapi_path} does not export aclnnAsStridedGetWorkspaceSize"
            )
        if not hasattr(self.opapi, "aclnnAsStrided"):
            raise RuntimeError(
                f"{opapi_path} does not export aclnnAsStrided"
            )

        print(f"[INFO] Using AsStrided opapi: {opapi_path}")

        try:
            check_acl(self.acl.aclInit(None), "aclInit")
            self.initialized = True

            check_acl(self.acl.aclrtSetDevice(device_id), "aclrtSetDevice")
            self.device_set = True

            check_acl(
                self.acl.aclrtCreateStream(ctypes.byref(self.stream)),
                "aclrtCreateStream",
            )
            self.stream_created = True
        except Exception:
            self.close()
            raise

    def _bind(self) -> None:
        void_pp = ctypes.POINTER(ctypes.c_void_p)
        int64_p = ctypes.POINTER(ctypes.c_int64)

        self.acl.aclInit.argtypes = [ctypes.c_char_p]
        self.acl.aclInit.restype = ctypes.c_int

        self.acl.aclFinalize.argtypes = []
        self.acl.aclFinalize.restype = ctypes.c_int

        self.acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
        self.acl.aclrtSetDevice.restype = ctypes.c_int

        self.acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
        self.acl.aclrtResetDevice.restype = ctypes.c_int

        self.acl.aclrtCreateStream.argtypes = [void_pp]
        self.acl.aclrtCreateStream.restype = ctypes.c_int

        self.acl.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
        self.acl.aclrtDestroyStream.restype = ctypes.c_int

        self.acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        self.acl.aclrtSynchronizeStream.restype = ctypes.c_int

        self.acl.aclrtMalloc.argtypes = [
            void_pp,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.acl.aclrtMalloc.restype = ctypes.c_int

        self.acl.aclrtFree.argtypes = [ctypes.c_void_p]
        self.acl.aclrtFree.restype = ctypes.c_int

        self.acl.aclrtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.acl.aclrtMemcpy.restype = ctypes.c_int

        self.nnopbase.aclCreateTensor.argtypes = [
            int64_p,
            ctypes.c_uint64,
            ctypes.c_int,
            int64_p,
            ctypes.c_int64,
            ctypes.c_int,
            int64_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.nnopbase.aclCreateTensor.restype = ctypes.c_void_p

        self.nnopbase.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self.nnopbase.aclDestroyTensor.restype = ctypes.c_int

        self.nnopbase.aclCreateIntArray.argtypes = [
            int64_p,
            ctypes.c_uint64,
        ]
        self.nnopbase.aclCreateIntArray.restype = ctypes.c_void_p

        self.nnopbase.aclDestroyIntArray.argtypes = [ctypes.c_void_p]
        self.nnopbase.aclDestroyIntArray.restype = ctypes.c_int

        self.opapi.aclnnAsStridedGetWorkspaceSize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            void_pp,
        ]
        self.opapi.aclnnAsStridedGetWorkspaceSize.restype = ctypes.c_int

        self.opapi.aclnnAsStrided.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.opapi.aclnnAsStrided.restype = ctypes.c_int

    @staticmethod
    def _int64_array(values: Sequence[int]):
        values = [int(v) for v in values]
        if not values:
            return None
        return (ctypes.c_int64 * len(values))(*values)

    def _malloc(self, size: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        check_acl(
            self.acl.aclrtMalloc(
                ctypes.byref(ptr),
                size,
                ACL_MEM_MALLOC_HUGE_FIRST,
            ),
            "aclrtMalloc",
        )
        return ptr

    def _create_tensor(
        self,
        host: np.ndarray,
        copy_to_device: bool,
    ) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        host = np.ascontiguousarray(host)

        acl_dtype = DTYPE_TO_ACL.get(host.dtype)
        if acl_dtype is None:
            raise TypeError(f"unsupported dtype: {host.dtype}")
        if host.nbytes == 0:
            raise ValueError("zero-sized tensor is not supported")

        device = self._malloc(host.nbytes)

        try:
            if copy_to_device:
                check_acl(
                    self.acl.aclrtMemcpy(
                        device,
                        host.nbytes,
                        ctypes.c_void_p(host.ctypes.data),
                        host.nbytes,
                        ACL_MEMCPY_HOST_TO_DEVICE,
                    ),
                    "aclrtMemcpy(H2D)",
                )

            shape = [int(v) for v in host.shape]
            strides = contiguous_strides(shape)

            tensor = self.nnopbase.aclCreateTensor(
                self._int64_array(shape),
                len(shape),
                acl_dtype,
                self._int64_array(strides),
                0,
                ACL_FORMAT_ND,
                self._int64_array(shape),
                len(shape),
                device,
            )

            if not tensor:
                raise RuntimeError("aclCreateTensor failed")

            return ctypes.c_void_p(tensor), device

        except Exception:
            self.acl.aclrtFree(device)
            raise

    def _create_int_array(self, values: Sequence[int]) -> ctypes.c_void_p:
        raw = self._int64_array(values)
        if raw is None:
            raise ValueError("aclIntArray cannot be empty")

        array = self.nnopbase.aclCreateIntArray(raw, len(values))
        if not array:
            raise RuntimeError("aclCreateIntArray failed")

        return ctypes.c_void_p(array)

    def run(
        self,
        input_x: np.ndarray,
        size: Sequence[int],
        stride: Sequence[int],
        storage_offset: int,
    ) -> np.ndarray:
        input_x = np.ascontiguousarray(input_x)
        size = [int(v) for v in size]
        stride = [int(v) for v in stride]

        if len(size) != len(stride):
            raise ValueError("size and stride must have the same rank")

        output = np.empty(tuple(size), dtype=input_x.dtype)

        tensors = []
        int_arrays = []
        buffers = []
        workspace = ctypes.c_void_p()

        try:
            input_tensor, input_dev = self._create_tensor(
                input_x, copy_to_device=True
            )
            output_tensor, output_dev = self._create_tensor(
                output, copy_to_device=False
            )

            tensors.extend([input_tensor, output_tensor])
            buffers.extend([input_dev, output_dev])

            size_array = self._create_int_array(size)
            stride_array = self._create_int_array(stride)
            int_arrays.extend([size_array, stride_array])

            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()

            check_acl(
                self.opapi.aclnnAsStridedGetWorkspaceSize(
                    input_tensor,
                    size_array,
                    stride_array,
                    int(storage_offset),
                    output_tensor,
                    ctypes.byref(workspace_size),
                    ctypes.byref(executor),
                ),
                "aclnnAsStridedGetWorkspaceSize",
            )

            if workspace_size.value:
                workspace = self._malloc(workspace_size.value)

            check_acl(
                self.opapi.aclnnAsStrided(
                    workspace,
                    workspace_size.value,
                    executor,
                    self.stream,
                ),
                "aclnnAsStrided",
            )

            check_acl(
                self.acl.aclrtSynchronizeStream(self.stream),
                "aclrtSynchronizeStream",
            )

            check_acl(
                self.acl.aclrtMemcpy(
                    ctypes.c_void_p(output.ctypes.data),
                    output.nbytes,
                    output_dev,
                    output.nbytes,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
                "aclrtMemcpy(D2H)",
            )

            return output

        finally:
            if workspace.value:
                self.acl.aclrtFree(workspace)

            for array in reversed(int_arrays):
                self.nnopbase.aclDestroyIntArray(array)

            for tensor in reversed(tensors):
                self.nnopbase.aclDestroyTensor(tensor)

            for buffer in reversed(buffers):
                self.acl.aclrtFree(buffer)

    def close(self) -> None:
        if self.stream_created:
            self.acl.aclrtDestroyStream(self.stream)
            self.stream_created = False

        if self.device_set:
            self.acl.aclrtResetDevice(self.device_id)
            self.device_set = False

        if self.initialized:
            self.acl.aclFinalize()
            self.initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


TEST_CASES = [
    ("T1_basic_1d", np.arange(10, dtype=np.float32), [5], [1], 0),
    ("T2_stride_2", np.arange(12, dtype=np.float32), [5], [2], 1),
    ("T3_negative_stride", np.arange(10, dtype=np.float32), [3], [-1], 9),
    (
        "T4_2d_contiguous",
        np.arange(12, dtype=np.float32).reshape(3, 4),
        [3, 4],
        [4, 1],
        0,
    ),
    (
        "T5_2d_non_contiguous",
        np.arange(20, dtype=np.float32),
        [3, 3],
        [4, 1],
        1,
    ),
    (
        "T6_2d_negative_inner_stride",
        np.arange(16, dtype=np.float32),
        [2, 3],
        [4, -1],
        3,
    ),
    ("T7_storage_offset", np.arange(10, dtype=np.float32), [4], [1], 3),
    ("T8_float16", np.arange(12, dtype=np.float16), [4], [2], 1),
    ("T9_int32", np.arange(12, dtype=np.int32), [4], [2], 2),
    (
        "T10_3d",
        np.arange(64, dtype=np.float32),
        [2, 2, 3],
        [12, 4, 1],
        1,
    ),
]


def compare(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, float]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False, float("inf")

    if np.issubdtype(actual.dtype, np.integer):
        diff = np.abs(
            actual.astype(np.int64) - expected.astype(np.int64)
        )
        return np.array_equal(actual, expected), float(np.max(diff))

    actual_f32 = actual.astype(np.float32)
    expected_f32 = expected.astype(np.float32)
    max_diff = float(np.max(np.abs(actual_f32 - expected_f32)))

    tol = 1e-3 if actual.dtype == np.float16 else 1e-5
    passed = np.allclose(
        actual_f32,
        expected_f32,
        rtol=tol,
        atol=tol,
        equal_nan=True,
    )
    return bool(passed), max_diff


def run_all_tests(runner: AclnnAsStrided) -> list[bool]:
    print("=" * 78)
    print("AsStrided custom-op accuracy test")
    print("NPU path: aclnnAsStridedGetWorkspaceSize -> aclnnAsStrided")
    print("=" * 78)

    results = []

    for name, input_x, size, stride, offset in TEST_CASES:
        try:
            expected = as_strided_golden(input_x, size, stride, offset)
            actual = runner.run(input_x, size, stride, offset)
            passed, max_diff = compare(actual, expected)
            results.append(passed)

            print(
                f"{name:<30} {'PASS' if passed else 'FAIL':<4} "
                f"dtype={str(input_x.dtype):<7} "
                f"size={str(size):<12} "
                f"stride={str(stride):<12} "
                f"offset={offset:<3} "
                f"max_diff={max_diff:.6e} "
                f"via=custom-aclnn"
            )

            if not passed:
                print(f"  golden: {expected}")
                print(f"  npu:    {actual}")

        except Exception as exc:
            results.append(False)
            print(f"{name:<30} ERROR")
            print(f"  {type(exc).__name__}: {exc}")

    print("-" * 78)
    print(f"Result: {sum(results)}/{len(results)} passed")
    print("=" * 78)

    return results


def main() -> int:
    device_id = int(os.environ.get("ASCEND_DEVICE_ID", "0"))

    try:
        with AclnnAsStrided(device_id=device_id) as runner:
            results = run_all_tests(runner)
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        return 1

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
