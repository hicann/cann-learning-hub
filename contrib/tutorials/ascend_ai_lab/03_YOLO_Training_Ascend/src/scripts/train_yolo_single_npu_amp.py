#!/usr/bin/env python3
"""Single-NPU entry point for the YOLO Ascend tuning lab."""

from train_yolo_ddp_amp import main


if __name__ == "__main__":
    main()
