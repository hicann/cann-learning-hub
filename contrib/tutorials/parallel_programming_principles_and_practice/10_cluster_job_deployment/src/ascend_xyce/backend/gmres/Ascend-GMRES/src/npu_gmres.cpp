#include "gmres.hpp"

namespace ascend_gmres {

GmresSolver make_host_prototype_gmres_solver() {
    return GmresSolver("HostPrototype optimized GMRES",
                       std::make_unique<HostPrototypePersistentBf16SpmvBackend>(),
                       std::make_unique<HostPrototypeBlasBackend>());
}

}  // namespace ascend_gmres
