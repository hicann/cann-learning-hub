/**
 * Ascend C 算子测试代码模板
 * 
 * 使用方法：
 * 1. 替换 "op" 为实际算子名
 * 2. 实现 ref_impl 参考函数
 * 3. 修改 test_case 中的参数和 API 调用
 * 4. 添加需要的测试用例
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <acl/acl.h>
#include "aclnn/acl_meta.h"
#include "aclnn_op.h"  // 替换为实际头文件

#define CHECK_ACL(ret) do { \
    if (ret != ACL_SUCCESS) { \
        std::cerr << "ACL 错误: " << ret << " 在第 " << __LINE__ << " 行" << std::endl; \
        return false; \
    } \
} while(0)

// ==================== 参考实现 ====================

template<typename T>
std::vector<T> ref_impl(const std::vector<T>& input, 
                        const std::vector<int64_t>& shape,
                        /* 其他参数 */) {
    std::vector<T> output(input.size());
    // TODO: 实现参考算法
    return output;
}

// ==================== 辅助函数 ====================

template<typename T>
bool compare_result(const std::vector<T>& output, const std::vector<T>& expected, 
                    float rtol = 1e-3, float atol = 1e-3) {
    if (output.size() != expected.size()) {
        std::cerr << "大小不匹配: " << output.size() << " vs " << expected.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs((float)output[i] - (float)expected[i]);
        float tolerance = atol + rtol * std::abs((float)expected[i]);
        if (diff > tolerance) {
            std::cerr << "不匹配在索引 " << i << ": 得到 " << (float)output[i] 
                      << ", 期望 " << (float)expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<int64_t> compute_stride(const std::vector<int64_t>& shape) {
    std::vector<int64_t> stride(shape.size());
    stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
}

int64_t compute_total(const std::vector<int64_t>& shape) {
    int64_t total = 1;
    for (auto d : shape) total *= d;
    return total;
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

// ==================== 测试用例模板 ====================

bool test_float32(const std::vector<int64_t>& shape, 
                  /* 参数 */, 
                  const std::string& case_name) {
    int64_t total = compute_total(shape);
    
    // 准备输入数据
    std::vector<float> input(total);
    for (int64_t i = 0; i < total; i++) input[i] = (float)(i % 100) / 10.0f;
    
    // 计算期望结果
    auto expected = ref_impl(input, shape, /* 参数 */);
    
    // 分配设备内存
    float* x_dev = nullptr;
    float* y_dev = nullptr;
    CHECK_ACL(aclrtMalloc((void**)&x_dev, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&y_dev, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(x_dev, total * sizeof(float), input.data(), 
                          total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    
    // 创建张量
    auto stride = compute_stride(shape);
    aclTensor* x_tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, 
                                          stride.data(), 0, ACL_FORMAT_ND, 
                                          shape.data(), shape.size(), x_dev);
    aclTensor* y_tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, 
                                          stride.data(), 0, ACL_FORMAT_ND, 
                                          shape.data(), shape.size(), y_dev);
    
    // TODO: 创建其他输入（如 axis）
    // int64_t axis_value = 1;
    // aclIntArray* axis_array = aclCreateIntArray(&axis_value, 1);
    
    // 调用算子 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnOpGetWorkspaceSize(x_tensor, /* 其他参数 */, y_tensor, 
                                      &workspaceSize, &executor));
    
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACL(aclnnOp(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // 获取结果
    std::vector<float> output(total);
    CHECK_ACL(aclrtMemcpy(output.data(), total * sizeof(float), y_dev, 
                          total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    
    // 验证结果
    bool pass = compare_result(output, expected);
    std::cout << "  " << case_name << " shape=" << shape_to_string(shape) 
              << ": " << (pass ? "✓" : "✗") << std::endl;
    
    // 清理资源
    if (workspace) CHECK_ACL(aclrtFree(workspace));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtFree(x_dev));
    CHECK_ACL(aclrtFree(y_dev));
    
    return pass;
}

// Float16 测试模板
bool test_float16(const std::vector<int64_t>& shape, 
                  /* 参数 */, 
                  const std::string& case_name) {
    int64_t total = compute_total(shape);
    
    std::vector<aclFloat16> input(total);
    for (int64_t i = 0; i < total; i++) input[i] = aclFloatToFloat16((float)(i % 100) / 10.0f);
    
    std::vector<float> input_f(total);
    for (int64_t i = 0; i < total; i++) input_f[i] = aclFloat16ToFloat(input[i]);
    auto expected_f = ref_impl(input_f, shape, /* 参数 */);
    
    aclFloat16* x_dev = nullptr;
    aclFloat16* y_dev = nullptr;
    CHECK_ACL(aclrtMalloc((void**)&x_dev, total * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&y_dev, total * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(x_dev, total * sizeof(aclFloat16), input.data(), 
                          total * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
    
    auto stride = compute_stride(shape);
    aclTensor* x_tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, 
                                          stride.data(), 0, ACL_FORMAT_ND, 
                                          shape.data(), shape.size(), x_dev);
    aclTensor* y_tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, 
                                          stride.data(), 0, ACL_FORMAT_ND, 
                                          shape.data(), shape.size(), y_dev);
    
    // TODO: 调用 API
    
    std::vector<aclFloat16> output(total);
    CHECK_ACL(aclrtMemcpy(output.data(), total * sizeof(aclFloat16), y_dev, 
                          total * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));
    
    std::vector<float> output_f(total);
    for (int64_t i = 0; i < total; i++) output_f[i] = aclFloat16ToFloat(output[i]);
    
    bool pass = compare_result(output_f, expected_f, 1e-2, 1e-2);
    std::cout << "  " << case_name << " shape=" << shape_to_string(shape) 
              << ": " << (pass ? "✓" : "✗") << std::endl;
    
    // 清理
    return pass;
}

// Int32 测试模板
bool test_int32(const std::vector<int64_t>& shape, 
                /* 参数 */, 
                const std::string& case_name) {
    // 类似 float32 实现
    return true;
}

// Int8 测试模板
bool test_int8(const std::vector<int64_t>& shape, 
               /* 参数 */, 
               const std::string& case_name) {
    // 类似 float32 实现
    return true;
}

// ==================== 主函数 ====================

int main() {
    // 初始化 ACL
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtContext context;
    CHECK_ACL(aclrtCreateContext(&context, 0));
    CHECK_ACL(aclrtSetCurrentContext(context));
    
    int passed = 0, failed = 0;
    
    // ========== 测试 1: 不同维度形状 ==========
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试 1: 不同维度形状" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (test_float32({10}, /* 参数 */, "1D")) passed++; else failed++;
    if (test_float32({5, 6}, /* 参数 */, "2D")) passed++; else failed++;
    if (test_float32({2, 3, 4}, /* 参数 */, "3D")) passed++; else failed++;
    if (test_float32({2, 3, 4, 5}, /* 参数 */, "4D")) passed++; else failed++;
    
    // ========== 测试 2: 大形状 ==========
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试 2: 大形状" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (test_float32({128, 256}, /* 参数 */, "2D_128x256")) passed++; else failed++;
    if (test_float32({16, 32, 64}, /* 参数 */, "3D_16x32x64")) passed++; else failed++;
    if (test_float32({64, 64, 64}, /* 参数 */, "3D_64x64x64")) passed++; else failed++;
    if (test_float32({1024, 1024}, /* 参数 */, "2D_1024x1024")) passed++; else failed++;
    
    // ========== 测试 3: 不同数据类型 ==========
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试 3: 不同数据类型" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (test_float32({2, 3, 4}, /* 参数 */, "float32")) passed++; else failed++;
    if (test_float16({2, 3, 4}, /* 参数 */, "float16")) passed++; else failed++;
    if (test_int32({2, 3, 4}, /* 参数 */, "int32")) passed++; else failed++;
    if (test_int8({2, 3, 4}, /* 参数 */, "int8")) passed++; else failed++;
    
    // ========== 测试 4: 边界情况 ==========
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试 4: 边界情况" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (test_float32({1, 5}, /* 参数 */, "dim1")) passed++; else failed++;
    if (test_float32({5, 1}, /* 参数 */, "dim1_axis1")) passed++; else failed++;
    
    // 清理
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    
    // 输出结果
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试结果汇总" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "通过: " << passed << std::endl;
    std::cout << "失败: " << failed << std::endl;
    std::cout << "总计: " << (passed + failed) << std::endl;
    std::cout << "========================================" << std::endl;
    
    return failed == 0 ? 0 : -1;
}
