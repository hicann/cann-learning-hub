rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 .. && make -j

# 生成测试输入数据
python3 ../scripts/gen_data.py --shape 7 83 --case-num 0 --dtype float
python3 ../scripts/gen_data.py --shape 1024 1024 --case-num 1 --dtype float16
python3 ../scripts/gen_data.py --shape 999 999 --case-num 2 --dtype float

# 执行编译生成的可执行程序，执行样例
./demo

# 验证输出结果是否正确，确认算法逻辑正确
python3 ../scripts/verify_result.py output/output_case0.bin output/golden_case0.bin float
python3 ../scripts/verify_result.py output/output_case1.bin output/golden_case1.bin float16
python3 ../scripts/verify_result.py output/output_case2.bin output/golden_case2.bin float