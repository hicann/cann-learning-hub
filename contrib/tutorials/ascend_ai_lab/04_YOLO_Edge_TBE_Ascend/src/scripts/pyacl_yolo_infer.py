#!/usr/bin/env python3
"""PyACL YOLO OM inference template for Atlas edge devices.

The script keeps the full application shape: preprocess -> OM inference ->
postprocess -> timing report. When PyACL or the OM file is unavailable, it runs
a dry-run with random model output so learners can still execute the notebook.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from benchmark_postprocess import postprocess_cpu
    from config_utils import load_config
except ImportError:
    from .benchmark_postprocess import postprocess_cpu
    from .config_utils import load_config


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    w, h = image.size
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, pad)
    arr = np.asarray(canvas, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)[None, ...].copy()
    return arr, scale, pad


def preprocess(image_path: str, cfg: dict) -> np.ndarray:
    size = int(cfg["preprocess"]["image_size"])
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8))
    data, _, _ = letterbox(image, size)
    mean = np.asarray(cfg["preprocess"].get("mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(cfg["preprocess"].get("std", [255.0, 255.0, 255.0]), dtype=np.float32).reshape(1, 3, 1, 1)
    return ((data - mean) / np.maximum(std, 1e-6)).astype(np.float32)


class PyAclYoloSession:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device_id = int(cfg["device"]["device_id"])
        self.om_path = Path(cfg["model"]["om_path"])
        self.acl = None
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers: list[tuple[int, int, object]] = []
        self.output_buffers: list[tuple[int, int, object]] = []
        self.output_sizes: list[int] = []
        self.device_set = False
        self.acl_inited = False
        self.dry_run = True

    def __enter__(self) -> "PyAclYoloSession":
        try:
            import acl

            self.acl = acl
            if not self.om_path.exists():
                print(f"[warning] OM model not found: {self.om_path}; using dry-run output")
                return self
            acl_config = self.cfg["device"].get("acl_json") or ""
            ret = acl.init(acl_config)

            if ret == 100002:
                print("[info] ACL already initialized, finalize and re-init")
                acl.finalize()
                ret = acl.init(acl_config)

            if ret != 0:
                raise RuntimeError(f"acl.init failed, ret={ret}")

            self.acl_inited = True
            ret = acl.rt.set_device(self.device_id)
            if ret != 0:
                raise RuntimeError(f"acl.rt.set_device failed, ret={ret}")
            self.device_set = True
            self.model_id, ret = acl.mdl.load_from_file(str(self.om_path))
            if ret != 0:
                raise RuntimeError(f"acl.mdl.load_from_file failed, ret={ret}")
            self.model_desc = acl.mdl.create_desc()
            ret = acl.mdl.get_desc(self.model_desc, self.model_id)
            if ret != 0:
                raise RuntimeError(f"acl.mdl.get_desc failed, ret={ret}")
            self._create_io_datasets()
            self.dry_run = False
            print(f"[OK] loaded OM model: {self.om_path}")
        except Exception as exc:
            print(f"[warning] PyACL path unavailable: {exc}")
            print("[warning] using dry-run output for code walkthrough")
            self._release()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release()

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.dry_run:
            shape = tuple(int(x) for x in self.cfg["model"]["output_shape"])
            rng = np.random.default_rng(2026)
            output = rng.random(shape, dtype=np.float32)
            output[..., :4] *= float(self.cfg["preprocess"]["image_size"])
            return output
        if self.acl is None or self.input_dataset is None or self.output_dataset is None:
            raise RuntimeError("PyACL session is not initialized")

        input_array = np.ascontiguousarray(input_tensor.astype(np.float32, copy=False))
        input_size = input_array.nbytes
        input_ptr, input_capacity, _ = self.input_buffers[0]
        if input_size > input_capacity:
            raise ValueError(f"input tensor uses {input_size} bytes, but OM input buffer has {input_capacity} bytes")

        self._memcpy(input_ptr, input_capacity, self._numpy_to_ptr(input_array), input_size, "host_to_device")
        ret = self.acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.execute failed, ret={ret}")

        raw_output = self._copy_output_to_bytes(0)
        expected_shape = tuple(int(x) for x in self.cfg["model"]["output_shape"])
        expected_values = int(np.prod(expected_shape))
        output = np.frombuffer(raw_output, dtype=np.float32)
        if output.size < expected_values:
            raise RuntimeError(
                f"OM output has {output.size} float32 values, but config expects {expected_values}; "
                "check model.output_shape in src/configs/yolo_edge.yaml"
            )
        return output[:expected_values].reshape(expected_shape).copy()

    def _create_io_datasets(self) -> None:
        assert self.acl is not None and self.model_desc is not None
        input_sizes = [
            int(self.acl.mdl.get_input_size_by_index(self.model_desc, i))
            for i in range(int(self.acl.mdl.get_num_inputs(self.model_desc)))
        ]
        self.output_sizes = [
            int(self.acl.mdl.get_output_size_by_index(self.model_desc, i))
            for i in range(int(self.acl.mdl.get_num_outputs(self.model_desc)))
        ]
        if not input_sizes or not self.output_sizes:
            raise RuntimeError("OM model must have at least one input and one output")
        self.input_dataset, self.input_buffers = self._make_dataset(input_sizes)
        self.output_dataset, self.output_buffers = self._make_dataset(self.output_sizes)

    def _make_dataset(self, sizes: list[int]) -> tuple[object, list[tuple[int, int, object]]]:
        assert self.acl is not None
        dataset = self.acl.mdl.create_dataset()
        buffers: list[tuple[int, int, object]] = []
        for size in sizes:
            ptr, ret = self.acl.rt.malloc(size, getattr(self.acl, "ACL_MEM_MALLOC_NORMAL_ONLY", 0))
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed, size={size}, ret={ret}")
            data_buffer = self.acl.create_data_buffer(ptr, size)
            ret = self._add_dataset_buffer(dataset, data_buffer)
            if ret != 0:
                raise RuntimeError(f"acl.mdl.add_dataset_buffer failed, ret={ret}")
            buffers.append((ptr, size, data_buffer))
        return dataset, buffers

    def _add_dataset_buffer(self, dataset: object, data_buffer: object) -> int:
        assert self.acl is not None
        result = self.acl.mdl.add_dataset_buffer(dataset, data_buffer)
        if isinstance(result, tuple):
            return int(result[-1])
        return int(result)

    def _numpy_to_ptr(self, array: np.ndarray) -> int:
        assert self.acl is not None
        util = getattr(self.acl, "util", None)
        if util is not None and hasattr(util, "numpy_to_ptr"):
            return util.numpy_to_ptr(array)
        return int(array.ctypes.data)

    def _memcpy(self, dst: int, dst_size: int, src: int, src_size: int, direction: str) -> None:
        assert self.acl is not None
        default_kind = 1 if direction == "host_to_device" else 2
        kind_name = "ACL_MEMCPY_HOST_TO_DEVICE" if direction == "host_to_device" else "ACL_MEMCPY_DEVICE_TO_HOST"
        ret = self.acl.rt.memcpy(dst, dst_size, src, src_size, getattr(self.acl, kind_name, default_kind))
        if ret != 0:
            raise RuntimeError(f"acl.rt.memcpy {direction} failed, ret={ret}")

    def _copy_output_to_bytes(self, output_index: int) -> bytes:
        assert self.acl is not None
        ptr, size, _ = self.output_buffers[output_index]
        host_ptr, ret = self.acl.rt.malloc_host(size)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc_host failed, size={size}, ret={ret}")
        try:
            self._memcpy(host_ptr, size, ptr, size, "device_to_host")
            util = getattr(self.acl, "util", None)
            if util is None or not hasattr(util, "ptr_to_bytes"):
                raise RuntimeError("acl.util.ptr_to_bytes is required to read model output")
            return util.ptr_to_bytes(host_ptr, size)
        finally:
            self.acl.rt.free_host(host_ptr)

    def _release_dataset(self, dataset: object | None, buffers: list[tuple[int, int, object]]) -> None:
        if self.acl is None:
            return
        for ptr, _, data_buffer in buffers:
            if data_buffer is not None:
                self.acl.destroy_data_buffer(data_buffer)
            if ptr:
                self.acl.rt.free(ptr)
        if dataset is not None:
            self.acl.mdl.destroy_dataset(dataset)
        buffers.clear()

    def _release(self) -> None:
        if self.acl is None:
            return
        self._release_dataset(self.input_dataset, self.input_buffers)
        self._release_dataset(self.output_dataset, self.output_buffers)
        self.input_dataset = None
        self.output_dataset = None
        if self.model_desc is not None:
            self.acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        if self.model_id is not None:
            self.acl.mdl.unload(self.model_id)
            self.model_id = None
        if self.device_set:
            self.acl.rt.reset_device(self.device_id)
            self.device_set = False
        if self.acl_inited:
            self.acl.finalize()
            self.acl_inited = False
        self.acl = None
        self.dry_run = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/yolo_edge.yaml")
    parser.add_argument("--image", default="")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_path = args.image or cfg.get("data", {}).get("sample_image", "")
    input_tensor = preprocess(image_path, cfg)

    with PyAclYoloSession(cfg) as session:
        for _ in range(3):
            pred = session.infer(input_tensor)
            _ = postprocess_cpu(pred, cfg)
        start = time.perf_counter()
        detections = None
        for _ in range(args.repeat):
            pred = session.infer(input_tensor)
            detections = postprocess_cpu(pred, cfg)
        latency = (time.perf_counter() - start) * 1000.0 / max(args.repeat, 1)

    print("=" * 72)
    print("PyACL YOLO inference report")
    print("=" * 72)
    print(f"image:       {image_path or '<synthetic gray image>'}")
    print(f"input shape: {input_tensor.shape}")
    print(f"detections:  {0 if detections is None else len(detections)}")
    print(f"latency:     {latency:.3f} ms / image")
    print("Use MindStudio Profiling to split model inference, data transfer, and postprocess time.")


if __name__ == "__main__":
    main()
