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
// NetOutput 故意在 Add 之前插入，验证拓扑排序确实遵循数据边而不是插入顺序。
Graph BuildAddGraph() {
  Node x{"x", "Data", {}, true, Tensor{"x", {1, 3, 224, 224}}};
  Node y{"y", "Data", {}, true, Tensor{"y", {1, 3, 224, 224}}};
  Node z{"add", "Add", {"x", "y"}, true, Tensor{"z", {1, 3, 224, 224}}};
  Node out{"net_output", "NetOutput", {"z"}, false, Tensor{}};
  Graph g("add_graph");
  g.Add(x).Add(y).Add(out).Add(z);
  return g;
}

bool HasExpectedStructure(const std::vector<Node>& nodes) {
  if (nodes.size() != 4) {
    return false;
  }
  std::map<std::string, const Node*> by_name;
  for (const auto& node : nodes) {
    by_name[node.name] = &node;
  }
  if (by_name.size() != 4 || by_name.count("x") == 0 || by_name.count("y") == 0 ||
      by_name.count("add") == 0 || by_name.count("net_output") == 0) {
    return false;
  }
  const Node& x = *by_name["x"];
  const Node& y = *by_name["y"];
  const Node& add = *by_name["add"];
  const Node& out = *by_name["net_output"];
  return x.op_type == "Data" && x.inputs.empty() && x.has_output && x.output.name == "x" &&
         y.op_type == "Data" && y.inputs.empty() && y.has_output && y.output.name == "y" &&
         add.op_type == "Add" && add.inputs == std::vector<std::string>{"x", "y"} &&
         add.has_output && add.output.name == "z" && out.op_type == "NetOutput" &&
         out.inputs == std::vector<std::string>{"z"} && !out.has_output;
}

int main() {
  Graph g = BuildAddGraph();
  const bool structure_ok = HasExpectedStructure(g.nodes());
  const std::vector<std::string> expected_order{"x", "y", "add", "net_output"};
  const auto topo_order = g.TopologicalOrder();
  bool topology_ok = topo_order.size() == expected_order.size();
  for (size_t index = 0; topology_ok && index < expected_order.size(); ++index) {
    topology_ok = topo_order[index].name == expected_order[index];
  }
  std::cout << "图: " << g.name() << ", 节点数: " << g.nodes().size() << "\n";
  std::cout << "拓扑序:\n";
  int i = 1;
  for (const auto& n : topo_order) {
    std::cout << "  " << i++ << ". " << n.name << " (" << n.op_type << ")\n";
  }
  std::cout << "是否动态 Shape 图: " << std::boolalpha << g.IsDynamicGraph() << "\n";

  // 把 x 的 batch 维改为动态(-1)，应判定为动态图；按名称查找，避免依赖插入顺序。
  const bool static_shape_ok = !g.IsDynamicGraph();
  bool dynamic_shape_ok = false;
  for (auto& node : g.nodes()) {
    if (node.name == "x" && node.has_output) {
      node.output.shape[0] = -1;
      dynamic_shape_ok = node.output.IsDynamic() && g.IsDynamicGraph();
      break;
    }
  }

  Graph cyclic("cyclic_graph");
  cyclic.Add(Node{"a", "Test", {"b"}, true, Tensor{"a", {1}}})
        .Add(Node{"b", "Test", {"a"}, true, Tensor{"b", {1}}});
  bool cycle_detection_ok = false;
  try {
    (void)cyclic.TopologicalOrder();
  } catch (const std::runtime_error&) {
    cycle_detection_ok = true;
  }

  const bool ok = structure_ok && topology_ok && cycle_detection_ok &&
                  static_shape_ok && dynamic_shape_ok;
  std::cout << (ok ? "\n[OK] 综合实践通过：图结构、拓扑排序、环检测及 Shape 判定正确。"
                   : "\n[FAIL] 请检查图结构、拓扑排序、环检测和 Shape 判定实现。")
            << std::endl;
  return ok ? 0 : 1;
}
