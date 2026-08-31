# 香橙派初始设置包

本部分用于设置香橙派启动自动播报IP地址的功能，**仅限于教师和助教进行操作**。

### 使用方式

本部分需要将香橙派通过HDMI线连接至外部显示屏上，并连接USB声卡和扬声器

1. 点击桌面右上角WiFi图标，连接学创WiFi（SSID：sic-guest）

2. 执行`git clone https://gitee.com/myronx/OrangePi-SIC`

3. 使用`pwd`指令查看本工程的路径是否为`/home/HwHiAiUser/OrangePi-SIC/00-StarterPack`

    ```bash
    # 设置学创WiFi、配置自动登录、配置关闭休眠
    bash 1.set_network.sh

    # 配置自动启动服务
    bash 2.install_services.sh
    ```

4. 安装完毕后，请使用`sudo reboot`指令重启板卡

5. 检查上电自动播报IP的功能是否生效。若正常，请将IP地址抄写在便利贴上后贴附在板卡的外壳上，以备后续使用。

配置完毕后，请不要随意改动本工程的位置。