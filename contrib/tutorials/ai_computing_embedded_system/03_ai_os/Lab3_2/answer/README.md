# answer/ 参考答案目录说明

本目录存放实验 3.2（昇腾香橙派 GPIO 驱动与 SPI 回环检测）的参考答案与预期输出，供实验完成后对照核查。

## 文件清单

| 文件 | 内容 |
|---|---|
| `README.md` | 参考答案总览与各任务核查要点 |
| `spi_loopback_output.md` | SPI 回环检测程序的预期输出（未短接 / 已短接两种情形） |
| `gpio_toggle_output.md` | GPIO 驱动测试程序的预期输出与示波器观测结果 |
| `lab_report_key_points.md` | 实验报告撰写参考要点（对应手册第四部分问题探究） |

## 任务依赖关系

```
任务一  SPI0 信道验证（系统工具 spidev_test，零依赖）
  ▼
任务二  配置 wiringOP 驱动库（一次性安装库）
  ▼
任务三  基于 wiringOP 的驱动程序（spi_loopback.c + gpio_toggle.c）
```

## 核查要点速览

### 任务一：SPI0 信道验证（系统工具，无需 wiringOP）
- `sudo spidev_test -D /dev/spidev0.0`：未短接时收发不一致；短接引脚 19 与 21 后收发一致。

### 任务二：wiringOP 安装
- `gpio readall` 输出引脚映射表，出现 SPI0_SD0(19)、SPI0_SDI(21)、SPI0_CLK(23)、SPI0_CS(24)。
- 注意：“配置 wiringOP”是装库（对所有引脚生效），不是给某个引脚做方向配置。

### 任务三：基于 wiringOP 的驱动程序
- `spi_loopback`：未短接显示 `[FAIL]`；短接引脚 19/21 后显示 `[PASS]`，通过率 100%。
- `gpio_toggle`：示波器在引脚 40（GPIO7_05 / wPi 25）观测到约 10µs 周期方波（约 45~48kHz），边沿清晰。
- `gpio_toggle 500000`：可见 LED 慢闪（0.5s 亮 0.5s 灭）。
- 关键：引脚 40 = GPIO7_05 = wPi 25（源码 `pinToGpio_AIPRO[25]=229=7*32+5`）；其方向由程序内 `pinMode(25, OUTPUT)` 在运行时配置，等价于 `gpio mode 25 out`，无需额外命令行配置。
