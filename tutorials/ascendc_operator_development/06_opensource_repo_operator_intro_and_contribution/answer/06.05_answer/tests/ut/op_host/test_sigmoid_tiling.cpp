#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class SigmoidTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SigmoidTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SigmoidTilingTest TearDown" << std::endl;
    }
};

TEST_F(SigmoidTilingTest, sigmoid_tiling_test_1)
{
    struct SigmoidCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "Sigmoid",
        {
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor1
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor2
        },
        {
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output tensor
        },
        {
            /* attrs */
        },
        &compileInfo,
        64,
        262144
        ); 
    uint64_t expectTilingKey = 0;
    string expectTilingData = "16384 2 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}