/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt_compile.h"

namespace {
bool CheckAcl(const char *operation, aclError ret) {
  if (ret == ACL_ERROR_NONE) {
    return true;
  }
  std::cerr << "[ERROR] " << operation << " failed, aclError=" << ret << std::endl;
  return false;
}

std::string ReadTextFile(const char *path) {
  std::ifstream input(path, std::ios::in);
  if (!input) {
    return {};
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

void PrintCompileLog(aclrtcProg prog) {
  size_t log_size = 0U;
  if ((aclrtcGetCompileLogSize(prog, &log_size) != ACL_ERROR_NONE) || (log_size == 0U)) {
    return;
  }
  std::vector<char> log(log_size, '\0');
  if (aclrtcGetCompileLog(prog, log.data()) == ACL_ERROR_NONE) {
    std::cerr << log.data() << std::endl;
  }
}
}  // namespace

int main(int argc, char *argv[]) {
  constexpr int kExpectedArgc = 4;
  if (argc != kExpectedArgc) {
    std::cerr << "Usage: " << argv[0] << " <kernel_source> <output_npubin> <soc_version>" << std::endl;
    return 1;
  }

  const std::string source = ReadTextFile(argv[1]);
  if (source.empty()) {
    std::cerr << "[ERROR] Failed to read kernel source: " << argv[1] << std::endl;
    return 1;
  }

  aclrtcProg prog = nullptr;
  if (!CheckAcl("aclrtcCreateProg", aclrtcCreateProg(&prog, source.c_str(), "add_custom", 0, nullptr, nullptr))) {
    return 1;
  }

  const std::string soc_option = std::string("--npu-soc=") + argv[3];
  const char *options[] = {soc_option.c_str()};
  if (!CheckAcl("aclrtcCompileProg", aclrtcCompileProg(prog, 1, options))) {
    PrintCompileLog(prog);
    aclrtcDestroyProg(&prog);
    return 1;
  }

  size_t binary_size = 0U;
  if (!CheckAcl("aclrtcGetBinDataSize", aclrtcGetBinDataSize(prog, &binary_size)) || (binary_size == 0U)) {
    aclrtcDestroyProg(&prog);
    return 1;
  }

  std::vector<char> binary(binary_size);
  if (!CheckAcl("aclrtcGetBinData", aclrtcGetBinData(prog, binary.data()))) {
    aclrtcDestroyProg(&prog);
    return 1;
  }

  std::ofstream output(argv[2], std::ios::out | std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "[ERROR] Failed to open output file: " << argv[2] << std::endl;
    aclrtcDestroyProg(&prog);
    return 1;
  }
  output.write(binary.data(), static_cast<std::streamsize>(binary.size()));
  output.close();

  if (!CheckAcl("aclrtcDestroyProg", aclrtcDestroyProg(&prog))) {
    return 1;
  }
  std::cout << "[INFO] Kernel compiled for " << argv[3] << ": " << argv[2] << " (" << binary_size << " bytes)"
            << std::endl;
  return 0;
}
