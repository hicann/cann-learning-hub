#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== 网络配置脚本 ===${NC}"

# 设置变量
CONNECTION_NAME="Wired Connection 1"
INTERFACE="eth0"
IP_ADDRESS="10.100.196.200/24"
GATEWAY="10.100.196.254"
DNS="10.96.0.100"

echo "配置信息："
echo "接口: $INTERFACE"
echo "IP地址: $IP_ADDRESS"
echo "网关: $GATEWAY"
echo "DNS: $DNS"
echo

# 检查NetworkManager是否运行
if ! systemctl is-active --quiet NetworkManager; then
    echo -e "${RED}NetworkManager未运行，正在启动...${NC}"
    sudo systemctl start NetworkManager
fi

# 删除现有连接（如果存在）
if nmcli con show | grep -q "$CONNECTION_NAME"; then
    echo "删除现有连接: $CONNECTION_NAME"
    sudo nmcli con delete "$CONNECTION_NAME"
fi

# 创建新连接
echo "创建新连接..."
sudo nmcli con add type ethernet con-name "$CONNECTION_NAME" ifname $INTERFACE \
    ipv4.addresses $IP_ADDRESS \
    ipv4.gateway $GATEWAY \
    ipv4.dns $DNS \
    ipv4.method manual

# 激活连接
echo "激活连接..."
sudo nmcli con up "$CONNECTION_NAME"

# 验证配置
echo -e "\n${GREEN}验证网络配置:${NC}"
ip addr show $INTERFACE
echo
echo -e "${GREEN}路由信息:${NC}"
ip route
echo
echo -e "${GREEN}DNS配置:${NC}"
cat /etc/resolv.conf

echo -e "\n${GREEN}网络配置完成！${NC}"