#include "csr_matrix.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    const std::string output_dir = argc > 1 ? argv[1] : "matrices";
    const std::string filter = argc > 2 ? argv[2] : "";
    std::filesystem::create_directories(output_dir);
    bool generated = false;
    for (const auto& spec : dis_gmres::default_matrix_specs()) {
      if (!filter.empty() && filter != spec.name) continue;
      auto matrix = dis_gmres::generate_matrix(spec, 42);
      const auto path = (std::filesystem::path(output_dir) / (spec.name + ".csrbin")).string();
      std::string error;
      if (!matrix.save_binary(path, &error)) throw std::runtime_error(error);
      std::cout << "saved " << path << " rows=" << matrix.rows << " cols=" << matrix.cols
                << " nnz=" << matrix.nnz() << '\n';
      generated = true;
    }
    if (!generated) throw std::runtime_error("unknown matrix filter: " + filter);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "matrix_generator: " << exception.what() << '\n';
    return 1;
  }
}
