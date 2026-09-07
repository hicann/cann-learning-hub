"""TileLang-style template for tiling YOLO NMS workloads.

This is a teaching template, not a ready-to-compile kernel. It documents the
tiling structure used when moving CPU postprocess hotspots onto the accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NmsTileConfig:
    num_boxes: int
    tile_boxes: int = 128
    vector_width: int = 32
    max_output: int = 300
    iou_threshold: float = 0.45

    @property
    def num_tiles(self) -> int:
        return (self.num_boxes + self.tile_boxes - 1) // self.tile_boxes


def describe_tile_schedule(cfg: NmsTileConfig) -> list[str]:
    return [
        f"split {cfg.num_boxes} candidate boxes into {cfg.num_tiles} tiles",
        f"load {cfg.tile_boxes} boxes per tile into local buffer",
        "broadcast selected box and compute IoU against one tile",
        "write suppression mask to workspace",
        "compact kept indices up to max_output",
    ]


def pseudo_tilelang_kernel() -> str:
    return """
@tilelang.jit
def yolo_nms_kernel(boxes, scores, keep, count, workspace):
    # 1. tile candidate boxes
    # 2. vectorize IoU computation
    # 3. update suppression mask
    # 4. compact selected indices
    pass
"""


if __name__ == "__main__":
    config = NmsTileConfig(num_boxes=25200)
    print("\n".join(describe_tile_schedule(config)))
    print(pseudo_tilelang_kernel())
