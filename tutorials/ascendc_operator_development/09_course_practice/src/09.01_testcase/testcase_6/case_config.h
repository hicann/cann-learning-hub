#pragma once

#include <cstdint>
#include <vector>

#include "acl/acl.h"

using HostDataType = uint16_t;
const std::vector<int64_t> CASE_SHAPE = {64, 64, 1024};
constexpr aclDataType CASE_ACL_DTYPE = aclDataType::ACL_BF16;
