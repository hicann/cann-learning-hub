if [ $# -eq 0 ]; then
    echo "错误：请提供 --npu-arch=xxx 参数"
    echo "用法: bash run.sh --npu-arch=dav-2201 或 bash run.sh --npu-arch=dav-3510"
    exit 1
fi

# 解析 --npu-arch=xxx 参数, 提取架构名
for arg in "$@"; do
    case "$arg" in
        --npu-arch=*)
            npu_arch="${arg#*=}"
            ;;
    esac
done

case "$npu_arch" in
    dav-2201|dav-3510)
        ;;
    *)
        echo "错误：不支持的架构名 '$npu_arch'，仅支持 dav-2201 或 dav-3510"
        exit 1
        ;;
esac

rm -rf build
mkdir -p build && cd build;                                               # 创建并进入build目录
cmake -DCMAKE_ASC_ARCHITECTURES="$npu_arch" ..;make -j;                   # 编译工程
python3 ../scripts/gen_data.py                                            # 生成输入数据、标杆产物
./mmad_leakyrelu                                                     # 执行可执行产物
python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
