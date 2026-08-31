# 实验3.2 昇腾香橙派 GPIO 驱动与 SPI 回环检测实验

> **实验平台**：Orange Pi AI Pro 开发板
> **副标题**：基于 Orange Pi AI Pro 的 GPIO 驱动程序与 SPI 回环检测实践

本实验基于 Orange Pi AI Pro 开发板，围绕 GPIO 驱动程序开发与 SPI 回环检测展开实践，通过 wiringOP 组件实现外设通讯与协议解析。

## Notebook

- [lab3.2_orange_pi_driver.ipynb](./lab3.2_orange_pi_driver.ipynb)

## 依赖说明

本实验依赖 **wiringOP** 库。开发板系统通常已在 `/usr/src` 下预装 wiringOP 源码，可直接编译安装；若系统中未预装，可通过以下任一方式获取：

**方式一：直接下载压缩包（推荐）**

```bash
wget -O wiringOP.zip https://www.qmpan.com/f/L1Jptr/wiringOP.zip
unzip wiringOP.zip
cd wiringOP
sudo ./build
```

**方式二：从源码仓库克隆**

```bash
git clone https://gitcode.com/orangepi/wiringOP.git
cd wiringOP
sudo ./build
```

> 仓库不再随附 `wiringOP.zip` 压缩包，请按上述方式获取并安装。
