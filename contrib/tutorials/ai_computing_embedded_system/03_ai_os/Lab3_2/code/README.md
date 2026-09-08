# code/ 目录说明

本目录存放实验 3.2（昇腾香橙派 GPIO 驱动与 SPI 回环检测）的完整 C 语言源码与编译脚本，所有程序均在**香橙派 Orange Pi AI Pro 开发板**上编译运行。

## 文件清单

| 文件 | 功能说明 |
|---|---|
| `gpio_toggle.c` | GPIO 驱动测试程序，将物理引脚 40（wPi 25）设为输出，以 10µs 周期翻转电平，用于示波器实测 GPIO 翻转速率与信号质量 |
| `spi_loopback.c` | SPI0 回环检测程序，发送 8 字节测试数据并同步接收比对，验证 MOSI/MISO 收发信道是否完好 |
| `Makefile` | 编译脚本，一键编译上述两个程序 |

## 前置条件

1. 已在香橙派上完成 wiringOP 驱动库的安装（`gpio readall` 可正常输出）；
2. SPI0 设备节点存在：`ls -l /dev/spidev0.0`。

## 编译

```bash
# 进入本目录
cd code

# 一键编译全部程序
make

# 或单独编译
make gpio    # 编译 gpio_toggle
make spi     # 编译 spi_loopback
```

## 运行

### 1. GPIO 驱动测试

```bash
# 10µs 周期翻转（理论 50kHz），用示波器在引脚 40 观察波形
sudo ./gpio_toggle

# 指定半周期（微秒），例如 500000us=0.5s，即可看到 LED 慢闪
sudo ./gpio_toggle 500000
```

### 2. SPI 回环检测

```bash
# 持续循环测试（每秒一次），短接引脚 19 与 21 后应显示 [PASS]
sudo ./spi_loopback

# 测试 5 次后退出并输出通过率统计
sudo ./spi_loopback 5
```

## 关键参数

| 参数 | 取值 | 说明 |
|---|---|---|
| GPIO 引脚 | 物理引脚 40 / wPi 25 | GPIO 输出测试引脚 |
| GPIO 半周期 | 10µs（默认） | 理论频率 50kHz，可用命令行参数调整 |
| SPI 通道 | 0 | SPI0 |
| SPI 速率 | 1000000 Hz（1MHz） | 回环测试速率，可按需修改源码宏 |

## 目录关系

- 实验说明：`../lab3.2_cann_orange_pi_driver.ipynb`
- 参考答案：`../answer/`
- 实验图片：`../images/`

> 本目录代码参考自实验3.1 的测试代码（`spi_loopback.c`、`gpio_10us.c`），在此基础上增加了命令行参数、通过率统计、可配置半周期与实时优先级封装。
