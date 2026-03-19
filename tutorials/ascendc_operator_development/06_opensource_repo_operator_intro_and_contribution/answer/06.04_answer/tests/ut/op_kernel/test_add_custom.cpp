#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_kernel/add_custom.cpp"


class AddCustomKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddCustomKernelTest SetUp\n" << std::endl;
        const std::string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./add_custom_data/");
    }
    static void TearDownTestCase()
    {
        std::cout << "AddCustomKernelTest TearDown\n" << std::endl;
    }
private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string AddCustomKernelTest::rootPath = "../../../../";
const std::string AddCustomKernelTest::dataPath = rootPath + "experimental/math/add_custom/tests/ut/op_kernel/add_custom_data";

TEST_F(AddCustomKernelTest, test_case_0)
{
    size_t xByteSize = 2 * 1024 * sizeof(float);
    size_t yByteSize = 2 * 1024 * sizeof(float);
    size_t zByteSize = 2 * 1024 * sizeof(float);
    size_t tiling_data_size = sizeof(AddCustomTilingData);
    uint32_t numBlocks = 64;

    system("cd ./add_custom_data/ && python3 gen_data.py");
    std::string input1 = "./add_custom_data/input1.bin";
    std::string input2 = "./add_custom_data/input2.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(input1, xByteSize, x, xByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    ReadFile(input2, yByteSize, y, xByteSize);

    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    std::string path(path_);

    AddCustomTilingData* tilingDatafromBin = reinterpret_cast<AddCustomTilingData*>(tiling);
    tilingDatafromBin->totalLength = 2 * 1024;
    tilingDatafromBin->tileNum = 1;


    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(add_custom<0>,
        numBlocks,
        x,
        y,
        z,
        workspace,
        (uint8_t *)(tilingDatafromBin));
    
    std::string outFile = "./add_custom_data/output.bin";
    WriteFile(outFile, z, zByteSize);

    // 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./add_custom_data/ && python3 compare_data.py");
    free(path_);
}
