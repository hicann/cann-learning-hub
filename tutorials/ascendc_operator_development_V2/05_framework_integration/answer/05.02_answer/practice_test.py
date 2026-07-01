import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.ops.load_library("libpractice_matmul_ops.so")


class TestPracticeMatmulAnswer(TestCase):
    def test_practice_matmul(self):
        torch.manual_seed(0)
        a = torch.rand([256, 64], device="cpu", dtype=torch.float16)
        b = torch.rand([64, 256], device="cpu", dtype=torch.float16)

        output = torch.ops.practice_matmul_ops.practice_matmul(a.npu(), b.npu()).cpu()
        expected = torch.matmul(a.float(), b.float()).half()
        self.assertRtolEqual(output, expected)


if __name__ == "__main__":
    run_tests()
