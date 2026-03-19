#include "../../../op_kernel/sigmoid.cpp"  
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>  
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

class SigmoidTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SigmoidTest SetUp\n" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./sigmoid_data/");
    }

    static void TearDownTestCase()
    {
        cout << "SigmoidTest TearDown\n" << endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string SigmoidTest::rootPath = "../../../../";  
const std::string SigmoidTest::dataPath = rootPath + "experimental/math/sigmoid/tests/ut/op_kernel/sigmoid_data";

TEST_F(SigmoidTest, test_case_0)
{
    size_t xByteSize = 8 * 2048 * sizeof(float);
    size_t yByteSize = 8 * 2048 * sizeof(float);
    size_t tiling_data_size = sizeof(SigmoidTilingData);
    uint32_t numBlocks = 8;

    system("cd ./sigmoid_data/ && python3 gen_data.py");
    std::string fileName = "./sigmoid_data/input.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(fileName, xByteSize, x, xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    SigmoidTilingData* tilingDatafromBin = reinterpret_cast<SigmoidTilingData*>(tiling);
    tilingDatafromBin->totalLength = 8 * 2048;
    tilingDatafromBin->tileNum = 2;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(sigmoid<0>,
        numBlocks,
        x,
        y,
        workspace,
        (uint8_t *)(tilingDatafromBin));
    
    fileName = "./sigmoid_data/output.bin";
    WriteFile(fileName, y, yByteSize);

    // 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./sigmoid_data/ && python3 compare_data.py");
    free(path_);
}
