#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace hccl_spmv {
struct CSRMatrix { int32_t rows=0, cols=0; std::vector<int32_t> row_ptr,col_idx; std::vector<float> values;
  int64_t nnz() const { return static_cast<int64_t>(values.size()); }
  bool validate(std::string* err=nullptr) const;
  bool save(const std::string& path,std::string* err=nullptr) const;
  static bool load(const std::string& path,CSRMatrix* out,std::string* err=nullptr);
};
CSRMatrix generate_matrix(const std::string& name);
std::vector<float> generate_x(int32_t cols);
void spmv_cpu(const CSRMatrix&,const std::vector<float>&,std::vector<float>*,int32_t first=0,int32_t last=-1);
double max_relative_error(const std::vector<float>&,const std::vector<float>&);
}
