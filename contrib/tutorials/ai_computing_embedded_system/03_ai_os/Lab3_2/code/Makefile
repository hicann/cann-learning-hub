# Makefile —— GPIO 驱动与 SPI 回环检测程序编译脚本
#
# 使用方法（在香橙派开发板上，进入 code/ 目录）：
#   make            # 编译全部程序
#   make gpio       # 仅编译 GPIO 驱动测试程序
#   make spi        # 仅编译 SPI 回环检测程序
#   make clean      # 清理编译产物
#
# 依赖：需先安装 wiringOP（见实验说明任务二）

CC      = gcc
CFLAGS  = -Wall -O2
LDFLAGS = -lwiringPi

TARGETS = gpio_toggle spi_loopback

all: $(TARGETS)

gpio: gpio_toggle.c
	$(CC) $(CFLAGS) -o gpio_toggle gpio_toggle.c $(LDFLAGS)

spi: spi_loopback.c
	$(CC) $(CFLAGS) -o spi_loopback spi_loopback.c $(LDFLAGS)

gpio_toggle: gpio_toggle.c
	$(CC) $(CFLAGS) -o gpio_toggle gpio_toggle.c $(LDFLAGS)

spi_loopback: spi_loopback.c
	$(CC) $(CFLAGS) -o spi_loopback spi_loopback.c $(LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all gpio spi clean
