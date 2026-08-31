"""ACL推理分类器模块

封装AscendCL (ACL) 接口, 实现OM模型的加载与推理。
推理流程: 初始化ACL -> 加载OM -> 分配内存 -> 执行推理 -> 结果回传 -> 资源释放
"""
import numpy as np
import acl

ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


class ACLClassifier:
    """基于AscendCL的OM模型推理分类器"""

    def __init__(self, om_path, device_id=0):
        self.om_path = om_path
        self.device_id = device_id
        self.initialized = False

    def init(self):
        """初始化ACL运行时: init -> set_device -> create_context"""
        acl.init()
        acl.rt.set_device(self.device_id)
        self.context, _ = acl.rt.create_context(self.device_id)
        self.initialized = True

    def load_model(self):
        """加载OM模型到Device内存, 查询输入输出规格并分配缓冲区"""
        with open(self.om_path, 'rb') as f:
            om_bytes = f.read()
        ptr = acl.util.bytes_to_ptr(om_bytes)
        self.model_id, _ = acl.mdl.load_from_mem(ptr, len(om_bytes))

        self.model_desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.model_desc, self.model_id)

        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)
        if self.output_size == 0:
            self.output_size = 10 * 4

        self.in_buf, _ = acl.rt.malloc(self.input_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        self.out_buf, _ = acl.rt.malloc(self.output_size, ACL_MEM_MALLOC_NORMAL_ONLY)

        self.in_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(
            self.in_ds, acl.create_data_buffer(self.in_buf, self.input_size))
        self.out_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(
            self.out_ds, acl.create_data_buffer(self.out_buf, self.output_size))

        self.stream, _ = acl.rt.create_stream()

    def infer(self, input_data):
        """执行单次推理

        Args:
            input_data: float32 numpy数组, 形状(1,1,28,28)

        Returns:
            output_np: float32 numpy数组, 模型输出logits
            elapsed_ms: 推理耗时(毫秒)
        """
        import time
        in_ptr = acl.util.numpy_to_ptr(input_data)
        acl.rt.memcpy(self.in_buf, self.input_size, in_ptr,
                      self.input_size, ACL_MEMCPY_HOST_TO_DEVICE)
        acl.rt.synchronize_stream(self.stream)

        t0 = time.perf_counter()
        acl.mdl.execute(self.model_id, self.in_ds, self.out_ds)
        acl.rt.synchronize_stream(self.stream)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        buf = acl.mdl.get_dataset_buffer(self.out_ds, 0)
        data = acl.get_data_buffer_addr(buf)
        actual_size = int(acl.get_data_buffer_size(buf)) or self.output_size
        num_out = actual_size // 4 if actual_size >= 4 else 10
        output_np = np.zeros(num_out, dtype=np.float32)
        out_ptr = acl.util.numpy_to_ptr(output_np)
        acl.rt.memcpy(out_ptr, actual_size, data,
                      actual_size, ACL_MEMCPY_DEVICE_TO_HOST)
        acl.rt.synchronize_stream(self.stream)

        return output_np, elapsed_ms

    def predict(self, input_data):
        """推理并返回预测类别和置信度"""
        output_np, elapsed_ms = self.infer(input_data)
        pred = int(output_np.argmax())
        confidence = float(np.exp(output_np[pred]) / np.exp(output_np).sum())
        return pred, confidence, elapsed_ms

    def release(self):
        """释放所有资源: stream -> dataset -> 内存 -> 模型 -> 上下文"""
        if not self.initialized:
            return
        acl.rt.destroy_stream(self.stream)
        acl.mdl.destroy_dataset(self.in_ds)
        acl.mdl.destroy_dataset(self.out_ds)
        acl.rt.free(self.in_buf)
        acl.rt.free(self.out_buf)
        acl.mdl.destroy_desc(self.model_desc)
        acl.mdl.unload(self.model_id)
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        self.initialized = False
