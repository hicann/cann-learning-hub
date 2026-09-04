# RoPE 源码

| 文件 | 作用 |
|---|---|
| `rope_simd_kernel.cpp` | 910B3 SIMD/Vector Core Kernel |
| `rope_simd_rtc.cpp` | RTC 编译、加载、执行、计时和 reference 校验 |
| `rope_simt_950.asc` | Ascend 950 SIMT 模板 |
| `run_rope_lab.sh` | 直接编译并运行 SIMD 实验 |

```bash
bash run_rope_lab.sh --warmup 2 --repeat 5
```

脚本使用 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME` 或常见安装目录寻找 CANN。成功标准是
`ROPE_RESULT status=PASS ... fallback=0 path=ASCENDC_SIMD_RTC device_id=0`。SIMT 模板必须在
Ascend 950 上另行编译、执行并与相同 reference 比较；910B3 结果不能替代该验证。
