// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(AsStrided,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32),
    ),
);
