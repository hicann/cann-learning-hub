#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class AddCustomInfershapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddCustomInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddCustomInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(AddCustomInfershapeTest, add_custom_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AddCustom",
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 1024},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
