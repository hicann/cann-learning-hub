#pragma once

#include <cstdint>
#include <vector>

#include "acl/acl.h"

using HostDataType = aclFloat16;
const std::vector<int64_t> CASE_SHAPE = {4, 2028};
constexpr aclDataType CASE_ACL_DTYPE = aclDataType::ACL_FLOAT16;
