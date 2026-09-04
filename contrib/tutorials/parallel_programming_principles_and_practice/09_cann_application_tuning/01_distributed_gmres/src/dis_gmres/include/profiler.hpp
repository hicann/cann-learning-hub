#pragma once

#include <cstddef>

namespace dis_gmres {

struct Profile {
  double total_ms = 0.0;
  double spmv_ms = 0.0;
  double dot_ms = 0.0;
  double axpy_ms = 0.0;
  double norm_ms = 0.0;
  double givens_ms = 0.0;
  double communication_ms = 0.0;
  double transfer_ms = 0.0;
  double kernel_launch_ms = 0.0;
  double synchronization_ms = 0.0;
  std::size_t allreduce_calls = 0;
  std::size_t allgather_calls = 0;

  void accumulate(const Profile& other);
  void scale(double factor);
};

}  // namespace dis_gmres
