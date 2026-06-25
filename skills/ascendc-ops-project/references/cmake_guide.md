# CMake 配置详解

## CMakePresets.json 详解

### 基本结构

```json
{
  "version": 3,
  "configurePresets": [...],
  "buildPresets": [...]
}
```

### 关键变量说明

| 变量 | 说明 | 必需 |
|------|------|------|
| `ENABLE_BINARY_PACKAGE` | 启用二进制打包 | 是 |
| `ASCEND_CANN_PACKAGE_PATH` | CANN 安装路径 | 是 |
| `vendor_name` | 厂商名称 | 是 |
| `ASCEND_PRODUCT_TYPE` | 目标芯片型号 | 否 |

### 多芯片支持

```json
{
  "cacheVariables": {
    "ASCEND_PRODUCT_TYPE": {
      "type": "STRING",
      "value": "ascend910b"
    }
  }
}
```

支持的芯片型号：
- `ascend910b` - Atlas 800 训算节点
- `ascend310p` - Atlas 200I A2
- `ascend910c` - Atlas 900 A2

## CMakeLists.txt 详解

### op_host CMakeLists.txt

```cmake
add_library({OPERATOR_NAME}_host SHARED
    {OPERATOR_NAME}.cpp
)

target_include_directories({OPERATOR_NAME}_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_link_libraries({OPERATOR_NAME}_host
    ${ASCEND_CANN_PACKAGE_PATH}/lib/libgraph.so
    ${ASCEND_CANN_PACKAGE_PATH}/lib/libge_runner.so
)
```

### op_kernel CMakeLists.txt

```cmake
add_library({OPERATOR_NAME}_kernel SHARED
    {OPERATOR_NAME}.cpp
)

target_include_directories({OPERATOR_NAME}_kernel PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

# 设置编译选项
target_compile_options({OPERATOR_NAME}_kernel PRIVATE
    -fPIC
    -fno-stack-protector
)
```

## 常见问题

### 1. 找不到 CANN

```
CMake Error: Could not find AscendC
```

**解决**：设置环境变量
```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

### 2. 编译选项不兼容

```
error: unrecognized command line option
```

**解决**：检查编译器版本，确保使用 GCC 7.3+

### 3. 链接错误

```
undefined reference to `...`
```

**解决**：检查 target_link_libraries 是否包含所有依赖库
