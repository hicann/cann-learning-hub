#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SigmoidInfershapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SigmoidInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SigmoidInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(SigmoidInfershapeTest, sigmoid_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Sigmoid",
        {
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 2048},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
