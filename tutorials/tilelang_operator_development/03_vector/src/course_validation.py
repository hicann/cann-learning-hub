"""CPU references and complete-output checks; no host-latency benchmarking."""
import torch, torch_npu
SEED = 20260911

def cpu_input(shape, case):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    if case == "random": return torch.randn(shape, generator=g, dtype=torch.float32)
    if case == "zero": return torch.zeros(shape, dtype=torch.float32)
    if case == "constant": return torch.full(shape, 2.0, dtype=torch.float32)
    if case == "ramp": return torch.linspace(-4, 4, int(torch.tensor(shape).prod())).reshape(shape)
    if case == "large": return torch.linspace(-1000, 1000, shape[-1]).expand(shape).clone()
    raise ValueError(case)

def checked_input(host, limit=2048):
    if host.dtype != torch.float32 or not host.is_contiguous():
        raise ValueError("Require contiguous fp32 input")
    if not torch.isfinite(host).all().item() or host.abs().max().item() > limit:
        raise ValueError("Require finite input within the declared magnitude limit")
    return host.to("npu")

def softmax_reference(host):
    mx = host.amax(1)
    shifted = host - mx[:, None]
    e = shifted.exp()
    sm = e.sum(1)
    return mx, shifted, e, sm, e / sm[:, None]

def check_softmax(outs, host, label):
    torch.npu.synchronize()
    actual = [o.cpu() for o in outs]
    refs = softmax_reference(host)
    for name, out, ref in zip(("amax", "subtract", "exp", "sum", "divide"), actual, refs):
        exact = name in ("amax", "subtract")
        torch.testing.assert_close(out, ref, rtol=0 if exact else 1e-5, atol=0 if exact else 1e-6)
        print(f"{label} {name}: full-output PASS; max_abs_error={(out-ref).abs().max().item():.8g}", flush=True)
    torch.testing.assert_close(actual[-1], torch.softmax(host, 1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[-1].sum(1), torch.ones(host.shape[0]), rtol=1e-5, atol=1e-6)
    assert torch.isfinite(actual[-1]).all().item() and (actual[-1] >= 0).all().item()
    print(f"{label}: torch.softmax, finite/nonnegative output and row-sum checks PASS", flush=True)
