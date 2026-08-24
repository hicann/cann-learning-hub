// 2.4 综合编程实践 参考答案 · 任务一：Add 图预期输出（纯 C++，无需 NPU）
// 编译运行：g++ -std=c++17 02.04_practice_answer.cpp -o add_baseline && ./add_baseline
// 任务二（离线编译四步骤）参考见同目录 02.04_practice_answer_task2.cpp。
#include <iostream>
#include <vector>

int main() {
  const std::vector<std::vector<float>> x{{1, 2, 3}, {4, 5, 6}};
  const std::vector<std::vector<float>> y{{10, 20, 30}, {40, 50, 60}};

  std::vector<std::vector<float>> expected(x.size(), std::vector<float>(x[0].size()));
  for (size_t i = 0; i < x.size(); ++i) {
    for (size_t j = 0; j < x[i].size(); ++j) {
      expected[i][j] = x[i][j] + y[i][j];  // Add 图语义：逐元素相加
    }
  }

  const std::vector<std::vector<float>> golden{{11, 22, 33}, {44, 55, 66}};
  std::cout << "预期输出（参考基准）:\n";
  for (const auto& row : expected) {
    for (float v : row) {
      std::cout << v << " ";
    }
    std::cout << "\n";
  }
  const bool ok = (expected == golden);
  std::cout << (ok ? "[OK] 预期输出已算好（作为核对 NPU 推理结果是否正确的参照）" : "[FAIL] 结果不符")
            << std::endl;
  return ok ? 0 : 1;
}
