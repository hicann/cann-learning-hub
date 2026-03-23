import os
import pickle
import torch
import torch_npu
import torchair
from torch._functorch.aot_autograd import aot_module_simplified
from torchair.configs.compiler_config import CompilerConfig


class MMWithCache(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


def custom_compiler(gm: torch.fx.GraphModule, example_inputs):
    cache_file = "./compiled_model.pkl"
    if os.path.exists(cache_file):
        print("发现缓存，正在加载...")
        with open(cache_file, "rb") as f:
            artifacts = pickle.load(f)
        from torchair.npu_fx_compiler import _CompiledFxGraph
        compiled_graph = _CompiledFxGraph.load_artifacts(artifacts)
        print("从缓存加载成功")
        return compiled_graph

    print("缓存未命中，开始编译...")
    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    compiler = torchair.get_compiler(compiler_config)
    compiled_graph = compiler(gm, example_inputs)
    artifacts = compiled_graph.dump_artifacts()
    with open(cache_file, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"编译产物已保存到 {cache_file}")
    return compiled_graph


def custom_backend_with_cache(gm: torch.fx.GraphModule, example_inputs):
    return aot_module_simplified(gm, example_inputs, fw_compiler=custom_compiler)


torch.npu.set_device(0)
x = torch.ones([2, 2], dtype=torch.int32).npu()
y = torch.ones([2, 2], dtype=torch.int32).npu()
model = torch.compile(
    MMWithCache().npu(),
    backend=custom_backend_with_cache,
    dynamic=False,
    fullgraph=True,
)
ret = model(x, y)
print(ret)