# 描述
本目录提供一个最小可用的PyTorch自定义AddRmsNorm算子；并在tutorials/llm_inference/qwen3_1.7B目录下Qwen模型的基础上，详细说明将该算子集成到Qwen3-1.7B模型的开发过程。

## 步骤1：AddRmsNorm算子开发：
相关代码请参考当前目录下的add_rms_norm_custom_bfloat16子目录。

功能说明、验证方法可参考：add_rms_norm_custom_bfloat16/README.md

完成开发后，务必按照add_rms_norm_custom_bfloat16/README.md中的说明执行构建、运行操作，确保libascendc_ops.so文件生成成功，单算子功能正常。

## 步骤2：启用模型的AddRmsNorm优化。
修改QWEN模型配置:

tutorials/llm_inference/qwen3_1.7B/src/inference_scripts/recipe_qwen3_1_7b/models/qwen/config/qwen3_1_7b_1tp.yaml
tutorials/llm_inference/qwen3_1.7B/src/inference_scripts/recipe_qwen3_1_7b/models/qwen/config/qwen3_1_7b_1tp_profile.yaml

将两个文件中的enable_npu_rmsnorm、enable_add_rmsnorm配置项均改为True，确保优化生效。

## 步骤3：将QWEN模型精度验证中的原生AddRmsNorm算子替换为自定义的AddRmsNorm算子
修改文件：
tutorials/llm_inference/qwen3_1.7B/src/qwen3_npu_adaptation.py

修改点（全部集中在run_add_rmsnorm_precision_check函数内部）：

1、加载自定义算子。
找到“import torch_npu”语句，并在其后加入如下操作(\<path\>替换成libascendc_ops.so的实际路径)：

torch.ops.load_library("\<path\>/libascendc_ops.so")

2、找到dtype_map定义，删除其中的float16、float32类型。（本例中自定义的AddRmsNorm算子只实现了bfloat16类型）

3、替换被测接口。

找到如下语句：

opt_norm, _, opt_sum = torch_npu.npu_add_rms_norm(residual, hidden_states, weight, epsilon)

将其替换为：

opt_norm, _, opt_sum = torch.ops.ascendc_ops.ascendc_add_rms_norm_custom(residual, hidden_states, weight, epsilon)

## 步骤4：将QWEN模型推理过程的原生AddRmsNorm算子替换为自定义的AddRmsNorm算子
修改文件：

tutorials/llm_inference/qwen3_1.7B/src/inference_scripts/recipe_qwen3_1_7b/models/qwen/models/modeling_qwen.py

1、加载自定义算子。

找到“import torch_npu”语句，并在其后加入如下操作(\<path\>替换成libascendc_ops.so的实际路径)：

torch.ops.load_library("\<path\>/libascendc_ops.so")

2、替换AddRmsNorm算子接口。

找到如下语句：

y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)

将其替换为：

y, _, x = torch.ops.ascendc_ops.ascendc_add_rms_norm_custom(residual, hidden_states, self.weight, self.variance_epsilon)

## 步骤5：算子替换后的精度验证及推理的执行脚本
将本目录下的op_custom_test.py脚本拷贝到tutorials/llm_inference/qwen3_1.7B/op_custom_test.py并执行。

**注意事项**

执行此脚本之前，需首先完成如下准备工作：

1、完成tutorials/llm_inference/qwen3_1.7B/目录下的基础课程学习，保证保证基线版本模型可以正常执行。

2、保证自定义算子已编译/验证通过。且"libascendc_ops.so"文件可正常访问。
