#pragma once

#include <cstdint>
#include <vector>

#include "acl/acl.h"

using HostDataType = float;
const std::vector<int64_t> CASE_SHAPE = {8, 16, 64};
constexpr aclDataType CASE_ACL_DTYPE = aclDataType::ACL_FLOAT;
