/*
 * spi_loopback.c —— SPI0 回环检测程序
 *
 * 功能说明：
 *   1. 基于 wiringOP 的 SPI 硬件控制接口，对香橙派 SPI0 信道做回环自检；
 *   2. 发送 8 字节递增测试数据（00 01 02 ... 07），同步接收并逐字节比对；
 *   3. 通过命令行参数可指定测试次数，默认持续循环每秒一次；
 *   4. 短接 MOSI（物理引脚 19）与 MISO（物理引脚 21）后，收发数据应完全
 *      一致，即回环检测通过；未短接时 MISO 悬空，接收为随机值，属正常现象。
 *
 * 硬件接线：
 *   • 回环短接：杜邦线连接物理引脚 19（MOSI）与引脚 21（MISO）；
 *   • SPI0 其余信号：SCLK=引脚23，CS=引脚24（系统默认已使能）。
 *
 * 编译运行（在香橙派开发板上）：
 *   gcc spi_loopback.c -o spi_loopback -lwiringPi
 *   sudo ./spi_loopback            # 持续循环测试
 *   sudo ./spi_loopback 5          # 测试 5 次后退出
 *
 * 文件位置：code/spi_loopback.c
 */

#include <wiringPiSPI.h>
#include <wiringPi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

/* SPI0 通道号（固定为 0） */
#define SPI_CHANNEL 0
/* SPI 速度（单位：Hz，最高支持 50MHz，回环测试推荐 1MHz） */
#define SPI_SPEED   1000000
/* 测试数据长度（字节） */
#define BUF_SIZE    8

int main(int argc, char *argv[])
{
    int max_count = 0;   /* 0 表示持续循环 */
    if (argc >= 2) {
        max_count = atoi(argv[1]);
    }

    /* 初始化 wiringPi */
    if (wiringPiSetup() == -1) {
        printf("wiringPi 初始化失败！\n");
        return 1;
    }

    /* 初始化 SPI0 */
    int fd = wiringPiSPISetup(SPI_CHANNEL, SPI_SPEED);
    if (fd == -1) {
        printf("SPI0 初始化失败！\n");
        return 1;
    }

    printf("SPI0 回环检测程序\n");
    printf("  通道号: %d\n", SPI_CHANNEL);
    printf("  速度:   %d Hz (%.1f MHz)\n", SPI_SPEED, (float)SPI_SPEED / 1000000);
    printf("  接线:   物理引脚19(MOSI) <-> 物理引脚21(MISO)\n");
    printf("  按 Ctrl+C 停止测试\n\n");

    /* 发送缓冲区：初始化测试数据 0x00,0x01,...,0x07 */
    uint8_t tx_buf[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++) {
        tx_buf[i] = (uint8_t)i;
    }

    int test_count = 0;
    int pass_count = 0;

    while (1) {
        /* 接收缓冲区：wiringPiSPIDataRW 会同时发送并接收，数据原地覆盖 */
        uint8_t rx_buf[BUF_SIZE];
        memcpy(rx_buf, tx_buf, BUF_SIZE);

        /* 执行 SPI 收发（核心函数） */
        wiringPiSPIDataRW(SPI_CHANNEL, rx_buf, BUF_SIZE);

        test_count++;
        printf("=== 第 %d 次测试 ===\n", test_count);
        printf("发送: ");
        for (int i = 0; i < BUF_SIZE; i++) {
            printf("%02X ", tx_buf[i]);
        }
        printf("\n接收: ");
        for (int i = 0; i < BUF_SIZE; i++) {
            printf("%02X ", rx_buf[i]);
        }

        /* 逐字节比对，判定回环是否通过 */
        int success = (memcmp(tx_buf, rx_buf, BUF_SIZE) == 0);
        if (success) {
            pass_count++;
            printf("  [PASS] 数据一致\n\n");
        } else {
            printf("  [FAIL] 数据不一致\n\n");
        }

        /* 达到指定测试次数则输出统计并退出 */
        if (max_count > 0 && test_count >= max_count) {
            printf("测试完成: 共 %d 次, 通过 %d 次, 通过率 %.1f%%\n",
                   test_count, pass_count,
                   100.0 * pass_count / test_count);
            break;
        }

        sleep(1);
    }

    return 0;
}
