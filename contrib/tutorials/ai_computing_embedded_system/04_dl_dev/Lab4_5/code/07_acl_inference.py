"""
07_acl_inference.py - 昇腾香橙派 ACL 推理
实验4.5 昇腾香橙派部署深度学习网络实验

本脚本在昇腾香橙派开发板上运行，使用 ACL (Ascend Computing Language)
加载 OM 离线模型进行推理。

ACL 是昇腾 AI 计算语言库，提供模型加载、推理等 C++/Python 接口。
香橙派 AIPro 搭载 Ascend 310B 芯片，适合端侧 AI 推理。

运行方式 (在香橙派上):
    python 07_acl_inference.py

前置条件:
    1. 香橙派已安装 CANN 工具链
    2. 已生成 OM 模型 (models/student_model.om)
    3. 测试图片在 images/ 目录下
"""

import os
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# 1. 图像预处理
# ============================================================
def preprocess_image(image_path):
    """预处理图片，返回模型输入张量

    与训练时的预处理保持一致:
    - Resize 到 224x224
    - ToTensor 转换为 CHW 格式并归一化到 [0,1]
    - Normalize 用 ImageNet 均值标准差归一化
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    # 增加 batch 维度: (3, 224, 224) -> (1, 3, 224, 224)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.numpy()


# ============================================================
# 2. ACL 推理 (使用 acl 库)
# ============================================================
class ACLModel:
    """封装昇腾 ACL OM 模型的完整推理生命周期

    一次性完成 acl 初始化、模型加载、输入/输出 dataset 创建；多次调用
    infer() 复用同一模型，避免重复 init/finalize 与模型重载。使用完毕
    后必须调用 close() 释放资源。

    CANN ACL Python 推理流程 (参考官方 sampleResnetQuickStart):
    - acl.rt.malloc / acl.rt.free              device 内存分配/释放
    - acl.create_data_buffer(ptr, size)        创建携带大小的 data buffer
    - acl.mdl.add_dataset_buffer(dataset, buf) 将 data buffer 加入 dataset
    - acl.rt.memcpy / malloc_host / free_host  host<->device 拷贝
    - acl.mdl.execute                         执行推理
    """

    # ACL_MEMCPY_HOST_TO_DEVICE = 1, ACL_MEMCPY_DEVICE_TO_HOST = 2
    _H2D = 1
    _D2H = 2
    # ACL_MEM_MALLOC_HUGE_FIRST = 0
    _MALLOC_POLICY = 0

    def __init__(self, om_model_path):
        import acl
        self.acl = acl

        self.model_id = None
        self.context = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffer = 0
        self.output_buffer = 0
        self.input_size = 0
        self.output_size = 0
        self._desc = None

        self._setup(om_model_path)

    def _setup(self, om_model_path):
        acl = self.acl

        ret = acl.init()
        assert ret == 0, f"acl.init failed: {ret}"
        ret = acl.rt.set_device(0)
        assert ret == 0, f"set_device failed: {ret}"
        self.context, ret = acl.rt.create_context(0)
        assert ret == 0, f"create_context failed: {ret}"

        self.model_id, ret = acl.mdl.load_from_file(om_model_path)
        assert ret == 0, f"load_from_file failed: {ret}"

        self._desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._desc, self.model_id)
        assert ret == 0, f"get_desc failed: {ret}"

        self.input_size = acl.mdl.get_input_size_by_index(self._desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self._desc, 0)

        # 输入 dataset: malloc device 内存 -> create_data_buffer -> add_dataset_buffer
        self.input_dataset = acl.mdl.create_dataset()
        self.input_buffer, ret = acl.rt.malloc(self.input_size, self._MALLOC_POLICY)
        assert ret == 0, f"malloc input failed: {ret}"
        input_buf = acl.create_data_buffer(self.input_buffer, self.input_size)
        self.input_dataset, ret = acl.mdl.add_dataset_buffer(
            self.input_dataset, input_buf)
        assert ret == 0, f"add input buffer failed: {ret}"

        # 输出 dataset
        self.output_dataset = acl.mdl.create_dataset()
        self.output_buffer, ret = acl.rt.malloc(self.output_size, self._MALLOC_POLICY)
        assert ret == 0, f"malloc output failed: {ret}"
        output_buf = acl.create_data_buffer(self.output_buffer, self.output_size)
        self.output_dataset, ret = acl.mdl.add_dataset_buffer(
            self.output_dataset, output_buf)
        assert ret == 0, f"add output buffer failed: {ret}"

    def infer(self, input_data):
        """执行一次推理

        Args:
            input_data: numpy 数组，形状 (1, 3, 224, 224)，float32
        Returns:
            (output_data, elapsed_ms): 输出 numpy 数组与推理耗时(ms)
        """
        acl = self.acl
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)

        # host -> device
        bytes_data = input_data.tobytes()
        host_ptr = acl.util.bytes_to_ptr(bytes_data)
        ret = acl.rt.memcpy(self.input_buffer, self.input_size,
                            host_ptr, input_data.nbytes, self._H2D)
        assert ret == 0, f"memcpy H2D failed: {ret}"

        # execute
        start = time.time()
        ret = acl.mdl.execute(self.model_id, self.input_dataset,
                              self.output_dataset)
        assert ret == 0, f"execute failed: {ret}"
        elapsed = (time.time() - start) * 1000

        # device -> host
        host_out, ret = acl.rt.malloc_host(self.output_size)
        assert ret == 0, f"malloc_host failed: {ret}"
        try:
            ret = acl.rt.memcpy(host_out, self.output_size,
                                self.output_buffer, self.output_size, self._D2H)
            assert ret == 0, f"memcpy D2H failed: {ret}"
            out_bytes = acl.util.ptr_to_bytes(host_out, self.output_size)
            output_data = np.frombuffer(out_bytes, dtype=np.float32).copy()
        finally:
            acl.rt.free_host(host_out)

        return output_data, elapsed

    def close(self):
        """释放 ACL 资源"""
        acl = self.acl
        if self.output_dataset is not None:
            acl.mdl.destroy_dataset(self.output_dataset)
        if self.input_dataset is not None:
            acl.mdl.destroy_dataset(self.input_dataset)
        if self.output_buffer:
            acl.rt.free(self.output_buffer)
        if self.input_buffer:
            acl.rt.free(self.input_buffer)
        if self._desc is not None:
            acl.mdl.destroy_desc(self._desc)
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
        if self.context is not None:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(0)
        acl.finalize()


def simulate_inference(input_data):
    """模拟推理（用于无 ACL 环境的流程演示）"""
    print("  [模拟] 生成随机输出用于流程演示")
    time.sleep(0.01)
    return np.random.randn(2).astype(np.float32)


# ============================================================
# 3. 主推理流程
# ============================================================
def main():
    print("=" * 60)
    print("步骤7: 昇腾香橙派 ACL 推理")
    print("=" * 60)

    om_model = './models/student_model.om'
    if not os.path.exists(om_model):
        print(f"警告: OM 模型 {om_model} 不存在")
        print("将使用模拟推理演示流程")

    # 测试图片
    test_images = [
        ('../images/cat1.jpg', 'cat', 0),
        ('../images/cat2.jpg', 'cat', 0),
        ('../images/dog1.jpg', 'dog', 1),
        ('../images/dog2.jpg', 'dog', 1),
    ]

    class_names = ['cat', 'dog']
    correct = 0
    total = 0

    print("\n开始推理...\n")
    print(f"{'图片':<20} {'真实标签':<10} {'预测标签':<10} {'置信度':<10} {'结果':<6}")
    print("-" * 60)

    # 初始化 ACL 模型 (仅一次，多张图片复用同一模型)
    acl_model = None
    if os.path.exists(om_model):
        try:
            acl_model = ACLModel(om_model)
        except ImportError:
            print("acl 模块未安装，使用模拟推理 (用于流程演示)")
            acl_model = None

    try:
        for img_path, true_name, true_label in test_images:
            if not os.path.exists(img_path):
                print(f"  跳过 {img_path} (文件不存在)")
                continue

            # 预处理
            input_data = preprocess_image(img_path)

            # ACL 推理
            if acl_model is not None:
                output, elapsed = acl_model.infer(input_data)
                print(f"  [ACL] 推理耗时: {elapsed:.2f} ms")
            else:
                output = simulate_inference(input_data)

            # 后处理: softmax + argmax
            output_exp = np.exp(output - output.max())
            probs = output_exp / output_exp.sum()
            pred_label = int(np.argmax(probs))
            confidence = float(probs[pred_label])

            pred_name = class_names[pred_label]
            is_correct = pred_label == true_label
            correct += is_correct
            total += 1

            status = "OK" if is_correct else "FAIL"
            print(f"{os.path.basename(img_path):<20} {true_name:<10} "
                  f"{pred_name:<10} {confidence:<10.4f} {status:<6}")
    finally:
        if acl_model is not None:
            acl_model.close()

    # 总结
    print("\n" + "=" * 60)
    if total > 0:
        print(f"推理完成: {correct}/{total} 正确, 准确率={correct/total*100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
