import os
import sys

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

sys.path.append(os.getcwd())
import ascendc_ops


class TestPybindCustomAdd(TestCase):
    def test_add_custom_ops(self):
        torch.manual_seed(0)
        shape = (8, 2048)
        x = torch.rand(shape, device="cpu", dtype=torch.float16)
        y = torch.rand(shape, device="cpu", dtype=torch.float16)

        output = ascendc_ops.ascendc_add(x.npu(), y.npu()).cpu()
        expected = torch.add(x, y)
        self.assertRtolEqual(output, expected)


if __name__ == "__main__":
    run_tests()
