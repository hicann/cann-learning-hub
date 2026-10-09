# 任务二参考答案：llm-datadist API 调用流程模拟与验证

from enum import Enum, auto


class LLMRole(Enum):
    PROMPT = auto()
    DECODER = auto()


class MockLLMDataDist:
    """模拟 llm-datadist"""
    def __init__(self):
        self.initialized = False
        self.linked = False
        self.finalized = False

    def init(self, options=""):
        self.initialized = True
        print("  [Mock] init() called")

    def link_clusters(self, clusters, timeout=5000):
        self.linked = True
        print(f"  [Mock] link_clusters({clusters}) called")

    def unlink_clusters(self, clusters, timeout=5000, force=True):
        self.linked = False
        print(f"  [Mock] unlink_clusters({clusters}) called")

    def finalize(self):
        self.finalized = True
        print("  [Mock] finalize() called")


class MockCacheManager:
    """模拟 CacheManager"""
    def __init__(self):
        self.registered = {}
        self.allocated = {}

    def register_blocks_cache(self, cache_key, mem_ptr, size):
        self.registered[cache_key] = (mem_ptr, size)
        print(f"  [Mock] register_blocks_cache({cache_key}) called")

    def allocate_cache(self, cache_desc):
        cache_id = f"cache_{len(self.allocated)}"
        self.allocated[cache_id] = cache_desc
        print(f"  [Mock] allocate_cache() -> {cache_id}")
        return cache_id

    def unregister_cache(self, cache_key):
        if cache_key in self.registered:
            del self.registered[cache_key]
            print(f"  [Mock] unregister_cache({cache_key}) called")

    def deallocate_cache(self, cache_id):
        if cache_id in self.allocated:
            del self.allocated[cache_id]
            print(f"  [Mock] deallocate_cache({cache_id}) called")


class FlowValidator:
    """验证 API 调用流程是否正确"""
    def __init__(self):
        self.call_sequence = []

    def record(self, step_name):
        self.call_sequence.append(step_name)

    def validate_prefill_push_flow(self):
        """验证 Prefill 端 Push 流程

        正确顺序：init → link → register → push → unlink → unregister → finalize
        """
        expected = ["init", "link", "register", "push", "unlink", "unregister", "finalize"]
        errors = []
        for i, expected_step in enumerate(expected):
            actual = self.call_sequence[i] if i < len(self.call_sequence) else "缺失"
            if actual != expected_step:
                errors.append(f"  步骤 {i+1}: 期望 '{expected_step}'，实际 '{actual}'")
        return errors


def simulate_prefill_push_flow():
    """模拟 Prefill 端 Push KV Cache 的完整流程"""
    validator = FlowValidator()
    datadist = MockLLMDataDist()
    cache_manager = MockCacheManager()
    decode_cluster = 1

    print("\n=== Prefill 端 Push 流程模拟 ===")

    # TODO-1: 初始化 LLMDataDist（角色为 PROMPT）
    datadist.init()
    validator.record("init")

    # TODO-2: 建立集群链接
    datadist.link_clusters([decode_cluster])
    validator.record("link")

    # TODO-3: 注册 KV Cache 内存
    cache_manager.register_blocks_cache("kv_cache_key", 0, 1073741824)
    cache_id = cache_manager.allocate_cache({"num_layers": 2})
    validator.record("register")

    # 模拟 Push 传输
    validator.record("push")
    print("  [Mock] push_blocks() called — KV Cache 已推送到 Decode 节点")

    # TODO-4: 资源释放（正确顺序：unlink → unregister → deallocate → finalize）
    datadist.unlink_clusters([decode_cluster])
    validator.record("unlink")

    cache_manager.unregister_cache("kv_cache_key")
    validator.record("unregister")

    cache_manager.deallocate_cache(cache_id)

    datadist.finalize()
    validator.record("finalize")

    # 验证流程
    errors = validator.validate_prefill_push_flow()
    if errors:
        print("\n❌ 流程验证失败：")
        for e in errors:
            print(e)
    else:
        print("\n✅ 流程验证通过！API 调用顺序正确。")


if __name__ == "__main__":
    simulate_prefill_push_flow()
