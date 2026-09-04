#include "gmres.hpp"

namespace ascend_gmres {

GmresSolver make_cpu_single_gmres_solver() {
    return GmresSolver("CPU single-thread GMRES",
                       std::make_unique<CpuSpmvBackend>(false),
                       std::make_unique<CpuBlasBackend>(false));
}

}  // namespace ascend_gmres
