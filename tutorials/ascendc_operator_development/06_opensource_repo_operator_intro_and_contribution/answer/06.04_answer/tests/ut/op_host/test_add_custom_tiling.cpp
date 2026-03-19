#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class AddCustomTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddCustomTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddCustomTilingTest TearDown" << std::endl;
    }
};

TEST_F(AddCustomTilingTest, add_custom_tiling_test_1)
{
    struct AddCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AddCustom",
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor1
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor2
        },
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output tensor
        },
        {
            /* attrs */
        },
        &compileInfo,
        64,
        262144
        ); 
    uint64_t expectTilingKey = 0;
    string expectTilingData = "2048 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}