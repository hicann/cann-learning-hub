import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.ops.load_library("libpractice_sub_ops.so")


class TestPracticeSubAnswer(TestCase):
    def test_practice_sub(self):
        torch.manual_seed(0)
        x = torch.rand([8, 2048], device="cpu", dtype=torch.float16)
        y = torch.rand([8, 2048], device="cpu", dtype=torch.float16)

        output = torch.ops.practice_sub_ops.practice_sub(x.npu(), y.npu()).cpu()
        expected = x - y
        self.assertRtolEqual(output, expected)


if __name__ == "__main__":
    run_tests()
