#ifndef GE_SAMPLE_UTILS_H_
#define GE_SAMPLE_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

#include "ge/ge_api.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

namespace sample {
constexpr uint32_t GRAPH_ID = 0U;
constexpr int64_t M = 8;
constexpr int64_t N = 2048;
constexpr size_t ELEMENT_COUNT = static_cast<size_t>(M * N);
constexpr float X_VALUE = 0.1F;
constexpr float Y_VALUE = 0.2F;
constexpr float DEFAULT_EXPECTED_VALUE = X_VALUE + Y_VALUE;
constexpr float EPSILON = 1e-3F;

inline ge::TensorDesc TensorDesc()
{
    return ge::TensorDesc(ge::Shape({M, N}), ge::FORMAT_ND, ge::DT_FLOAT);
}

inline bool BuildInput(float value, ge::Tensor &tensor)
{
    tensor = ge::Tensor(TensorDesc());
    std::vector<float> data(ELEMENT_COUNT);
    std::fill(data.begin(), data.end(), value);
    const auto ret = tensor.SetData(reinterpret_cast<const uint8_t *>(data.data()),
                                    data.size() * sizeof(float));
    if (ret != ge::GRAPH_SUCCESS) {
        std::cerr << "Tensor::SetData failed, ret: " << ret << std::endl;
        return false;
    }
    return true;
}

inline bool CheckOutput(const ge::Tensor &tensor, float expected, const char *case_name)
{
    if (tensor.GetSize() < ELEMENT_COUNT * sizeof(float)) {
        return false;
    }
    const auto *data = reinterpret_cast<const float *>(tensor.GetData());
    for (size_t i = 0; i < ELEMENT_COUNT; ++i) {
        if (std::abs(data[i] - expected) > EPSILON) {
            std::cerr << "Mismatch at " << i << ": expected " << expected
                      << ", got " << data[i] << std::endl;
            return false;
        }
    }
    std::cout << case_name << " = " << data[0]
              << ", shape = [" << M << ", " << N << "]" << std::endl;
    return true;
}

inline int Run(const ge::Graph &graph, float expected = DEFAULT_EXPECTED_VALUE,
               float x_value = X_VALUE, float y_value = Y_VALUE,
               const char *case_name = "0.1 + 0.2")
{
    const char *device_id = std::getenv("ASCEND_DEVICE_ID");
    if (device_id == nullptr) {
        device_id = "0";
    }
    std::map<ge::AscendString, ge::AscendString> options = {
        {"ge.exec.deviceId", device_id},
        {"ge.graphRunMode", "1"},
    };
    const auto init_ret = ge::GEInitialize(options);
    if (init_ret != ge::SUCCESS) {
        std::cerr << "GEInitialize failed, ret: " << init_ret << std::endl;
        return 1;
    }

    int result = 1;
    {
        ge::Session session(options);
        const auto add_graph_ret = session.AddGraph(GRAPH_ID, graph);
        if (add_graph_ret != ge::SUCCESS) {
            std::cerr << "Session::AddGraph failed, ret: " << add_graph_ret << std::endl;
        } else {
            ge::Tensor x;
            ge::Tensor y;
            if (!BuildInput(x_value, x) || !BuildInput(y_value, y)) {
                std::cerr << "Build input tensors failed." << std::endl;
            } else {
                std::vector<ge::Tensor> outputs;
                const auto run_ret = session.RunGraph(GRAPH_ID, {x, y}, outputs);
                if (run_ret != ge::SUCCESS) {
                    std::cerr << "Session::RunGraph failed, ret: " << run_ret << std::endl;
                } else if (outputs.size() != 1U) {
                    std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
                } else if (!CheckOutput(outputs[0], expected, case_name)) {
                    std::cerr << "Output verification failed." << std::endl;
                } else {
                    std::cout << "GE sample success." << std::endl;
                    result = 0;
                }
            }
        }
    }
    (void)ge::GEFinalize();
    return result;
}
}  // namespace sample

#endif  // GE_SAMPLE_UTILS_H_
