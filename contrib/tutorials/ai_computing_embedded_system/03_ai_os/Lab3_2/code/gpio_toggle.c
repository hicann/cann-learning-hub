/*
 * gpio_toggle.c —— GPIO 驱动测试程序
 *
 * 功能说明：
 *   1. 基于 wiringOP 库在香橙派 Orange Pi AI Pro 上完成 GPIO 输出驱动测试；
 *   2. 将物理引脚 40（wPi 编号 25）设置为输出模式，以 10us 周期翻转电平，
 *      用于示波器实测 GPIO 翻转速率与信号质量；
 *   3. 通过提升进程优先级（SCHED_FIFO + nice=-20）减少系统调度抖动，
 *      保证波形周期稳定；
 *   4. 若将 delayMicroseconds 参数加大（如 500000），同一套 GPIO 电平翻转
 *      机制即可降速为 LED 灯闪烁效果，体现“一套底层机制、多种上层应用”。
 *
 * 硬件接线：
 *   • 示波器探头接物理引脚 40（wPi 25），GND 接开发板任意 GND 引脚；
 *   • 若做 LED 闪烁，在引脚 40 与 GND 之间串接 LED + 330Ω 限流电阻。
 *
 * 编译运行（在香橙派开发板上）：
 *   gcc gpio_toggle.c -o gpio_toggle -lwiringPi
 *   sudo ./gpio_toggle
 *
 * 文件位置：code/gpio_toggle.c
 */

#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

/* 物理引脚 40 = wPi 编号 25 = GPIO7_05（GPIO 全局编号 229 = 7*32+5） */
#define PIN             25
/* 默认翻转半周期（微秒），10us 对应理论 50kHz */
#define DEFAULT_HALF_US 10

/* 提升进程优先级，减少调度抖动 */
static void set_realtime_priority(void)
{
    setpriority(PRIO_PROCESS, 0, -20);
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        perror("sched_setscheduler");
    }
}

int main(int argc, char *argv[])
{
    int half_us = DEFAULT_HALF_US;

    /* 命令行可指定半周期（微秒），方便切换“高速翻转”与“LED 闪烁” */
    if (argc >= 2) {
        half_us = atoi(argv[1]);
        if (half_us <= 0) {
            half_us = DEFAULT_HALF_US;
        }
    }

    /* 初始化 wiringOP（wPi 编号体系） */
    if (wiringPiSetup() == -1) {
        printf("wiringPi 初始化失败！\n");
        return 1;
    }

    /* 设置引脚为输出模式 */
    pinMode(PIN, OUTPUT);

    /* 提升实时优先级 */
    set_realtime_priority();

    printf("GPIO 驱动测试程序\n");
    printf("  物理引脚: 40 (wPi %d)\n", PIN);
    printf("  半周期:   %d us\n", half_us);
    printf("  理论频率: %.2f kHz\n", 1000.0 / (2.0 * half_us));
    printf("  按 Ctrl+C 停止\n\n");

    /* 主循环：周期性翻转电平 */
    while (1) {
        digitalWrite(PIN, HIGH);
        delayMicroseconds(half_us);

        digitalWrite(PIN, LOW);
        delayMicroseconds(half_us);
    }

    return 0;
}
