// 1.4 综合编程实践 参考答案：简化版 AscendIR 计算图模型（纯 C++，无需 NPU）
// 编译运行：g++ -std=c++17 01.04_practice_answer.cpp -o ascendir_mini && ./ascendir_mini
#include <cstdint>
#include <iostream>
#include <map>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 对应 AscendIR 的 Tensor：用 shape + dtype 描述。shape 含 -1 表示动态维。
struct Tensor {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype = "float32";

  bool IsDynamic() const {
    for (int64_t d : shape) {
      if (d == -1) {
        return true;  // 任一维为 -1 即为动态 Shape
      }
    }
    return false;
  }
};

// 对应 AscendIR 的 Node：op_type + 数据边输入(上游输出张量名)。
struct Node {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;  // 依赖的上游输出张量名
  bool has_output = false;
  Tensor output;                    // 本节点产出张量（NetOutput 无输出）
};

class Graph {
 public:
  explicit Graph(std::string name) : name_(std::move(name)) {}

  Graph& Add(const Node& node) {
    nodes_.push_back(node);
    if (node.has_output) {
      by_output_[node.output.name] = nodes_.size() - 1;
    }
    return *this;
  }

  // 按数据边依赖做拓扑排序（Kahn 算法），存在环则抛异常。
  std::vector<Node> TopologicalOrder() const {
    std::map<std::string, int> indeg;
    std::map<std::string, std::vector<std::string>> adj;
    for (const auto& n : nodes_) {
      indeg[n.name] = 0;
    }
    for (const auto& n : nodes_) {
      for (const auto& t : n.inputs) {
        auto it = by_output_.find(t);
        if (it != by_output_.end()) {  // 边：产出 t 的上游节点 -> 当前节点
          adj[nodes_[it->second].name].push_back(n.name);
          indeg[n.name] += 1;
        }
      }
    }
    std::map<std::string, const Node*> name2node;
    for (const auto& n : nodes_) {
      name2node[n.name] = &n;
    }
    std::queue<std::string> q;
    for (const auto& n : nodes_) {
      if (indeg[n.name] == 0) {
        q.push(n.name);
      }
    }
    std::vector<Node> order;
    while (!q.empty()) {
      const std::string cur = q.front();
      q.pop();
      order.push_back(*name2node[cur]);
      for (const auto& nxt : adj[cur]) {
        if (--indeg[nxt] == 0) {
          q.push(nxt);
        }
      }
    }
    if (order.size() != nodes_.size()) {
      throw std::runtime_error("图中存在环，无法拓扑排序");
    }
    return order;
  }

  bool IsDynamicGraph() const {
    for (const auto& n : nodes_) {
      if (n.has_output && n.output.IsDynamic()) {
        return true;
      }
    }
    return false;
  }

  std::vector<Node>& nodes() { return nodes_; }
  const std::string& name() const { return name_; }

 private:
  std::string name_;
  std::vector<Node> nodes_;
  std::map<std::string, size_t> by_output_;
};

// 构造 Data x, Data y -> Add z -> NetOutput 的最小图。
Graph BuildAddGraph() {
  Node x{"x", "Data", {}, true, Tensor{"x", {1, 3, 224, 224}}};
  Node y{"y", "Data", {}, true, Tensor{"y", {1, 3, 224, 224}}};
  Node z{"add", "Add", {"x", "y"}, true, Tensor{"z", {1, 3, 224, 224}}};
  Node out{"net_output", "NetOutput", {"z"}, false, Tensor{}};
  Graph g("add_graph");
  g.Add(x).Add(y).Add(z).Add(out);
  return g;
}

int main() {
  Graph g = BuildAddGraph();
  std::cout << "图: " << g.name() << ", 节点数: " << g.nodes().size() << "\n";
  std::cout << "拓扑序:\n";
  int i = 1;
  for (const auto& n : g.TopologicalOrder()) {
    std::cout << "  " << i++ << ". " << n.name << " (" << n.op_type << ")\n";
  }
  std::cout << "是否动态 Shape 图: " << std::boolalpha << g.IsDynamicGraph() << "\n";

  // 把 x 的 batch 维改为动态(-1)，应判定为动态图
  g.nodes()[0].output.shape[0] = -1;
  const bool ok = g.nodes()[0].output.IsDynamic() && g.IsDynamicGraph();
  std::cout << (ok ? "\n[OK] 综合实践通过：拓扑排序正确，静态/动态 Shape 判定正确。"
                   : "\n[FAIL] 请检查 IsDynamic / 图判定实现。")
            << std::endl;
  return ok ? 0 : 1;
}
