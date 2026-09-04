#include "gmres.hpp"

namespace ascend_gmres {

GmresSolver make_cpu_openmp16_gmres_solver() {
    return GmresSolver("CPU OpenMP16 GMRES",
                       std::make_unique<CpuSpmvBackend>(true),
                       std::make_unique<CpuBlasBackend>(true));
}

}  // namespace ascend_gmres
