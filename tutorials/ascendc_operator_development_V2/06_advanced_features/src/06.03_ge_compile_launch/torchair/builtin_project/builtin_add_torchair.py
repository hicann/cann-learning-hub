import torch
import torch_npu
import torchair


class BuiltinAddModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def main():
    shape = (8, 2048)
    torch.manual_seed(0)
    x_cpu = torch.rand(shape, device="cpu", dtype=torch.float32)
    y_cpu = torch.rand(shape, device="cpu", dtype=torch.float32)

    config = torchair.CompilerConfig()
    backend = torchair.get_npu_backend(compiler_config=config)
    model = BuiltinAddModel().npu()
    opt_model = torch.compile(model, fullgraph=True, backend=backend, dynamic=False)

    output = opt_model(x_cpu.npu(), y_cpu.npu()).cpu()
    golden = torch.add(x_cpu, y_cpu)
    torch.testing.assert_close(output, golden, rtol=1e-4, atol=1e-4)
    print("TorchAir Built-in Add sample success.")


if __name__ == "__main__":
    main()
