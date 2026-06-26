// 2.4 综合编程实践 参考答案 · 任务二：离线编译四步骤
// Initialize -> BuildModel -> SaveModel -> Finalize，产出 add_sample.om。
// 这是完整可用的源文件（与本章 02.02「动手实践」的 build_add_model.cpp 一致），
// 依赖 CANN 头文件与 GE 库，需在昇腾环境编译：把本文件放入 02.02 落盘的
// Sources/01.04/ 工程替换 build_add_model.cpp 后 cmake 编译运行；无卡环境不编译。
#include "common.h"
#include "ge/ge_ir_build.h"
#include <iostream>
#include <map>
#include <string>

namespace {
std::map<ge::AscendString, ge::AscendString> GlobalOptionsWithSoc(const std::string &soc_version) {
  std::map<ge::AscendString, ge::AscendString> opts;
  if (!soc_version.empty()) {  // 无卡编译时通过 ge.socVersion 指定目标芯片
    opts.emplace(ge::AscendString("ge.socVersion"), ge::AscendString(soc_version.c_str()));
  }
  return opts;
}
}  // namespace

int main(int argc, char **argv) {
  std::string soc_version;
  if (argc >= 3 && std::string(argv[1]) == "--soc-version") {
    soc_version = argv[2];
  }

  auto global_options = GlobalOptionsWithSoc(soc_version);
  if (ge::aclgrphBuildInitialize(global_options) != ge::GRAPH_SUCCESS) {  // 步骤 1：初始化编译环境
    std::cerr << "[Error] aclgrphBuildInitialize 失败\n";
    return -1;
  }
  std::cout << "[Info] 系统初始化成功\n";

  auto graph = MakeOfflineAddGraph();  // ES 构图得到 Add 图
  ge::ModelBufferData model;
  std::map<ge::AscendString, ge::AscendString> build_options;
  build_options.emplace(ge::AscendString("input_format"), ge::AscendString("ND"));  // Add 图输入为 ND
  if (ge::aclgrphBuildModel(*graph, build_options, model) != ge::GRAPH_SUCCESS) {    // 步骤 2：编译为内存模型
    std::cerr << "[Error] aclgrphBuildModel 失败\n";
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[Info] 模型构建成功，模型大小: " << model.length << " bytes\n";

  if (ge::aclgrphSaveModel("add_sample", model) != ge::GRAPH_SUCCESS) {  // 步骤 3：序列化保存为 add_sample.om
    std::cerr << "[Error] aclgrphSaveModel 失败\n";
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[Info] 模型保存成功，add_sample.om 已生成在当前目录。\n";

  ge::aclgrphBuildFinalize();  // 步骤 4：释放编译资源
  std::cout << "[Info] 系统释放成功\n";
  return 0;
}
