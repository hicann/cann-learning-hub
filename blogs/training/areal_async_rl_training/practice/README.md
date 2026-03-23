# AReaL 昇腾实践手册

本实践将呈现如何使用AReaL框架在昇腾设备进行强化学习训练.

## 1. 环境准备
|依赖  |版本要求  |  
|--|--|
| 硬件 | A2、A3系列硬件(双卡或以上) |
|镜像| areal_npu 0.5.0+|
## 2. 拉取官方 NPU 镜像

```bash
docker pull swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v0.5.0-a3
```

Atlas A2 设备请替换为对应的 `a2` 镜像标签。

## 3. 启动容器

下面的命令负责挂载 Ascend 设备、驱动和工作目录。执行前请把路径改成自己的真实路径，并按机器实际卡数调整 `--device`。

```bash
WORK_DIR=/path/to/your/workspace
CONTAINER_WORK_DIR=/workspace
CONTAINER_NAME=areal_npu
IMAGE=swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v0.5.0-a3

docker run -itd --cap-add=SYS_PTRACE --net=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci8 \
--device=/dev/davinci9 \
--device=/dev/davinci10 \
--device=/dev/davinci11 \
--device=/dev/davinci12 \
--device=/dev/davinci13 \
--device=/dev/davinci14 \
--device=/dev/davinci15 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--shm-size=1200g \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v ${WORK_DIR}:${CONTAINER_WORK_DIR} \
--privileged=true \
--name ${CONTAINER_NAME} \
${IMAGE} \
/bin/bash
```

## 4. 安装 AReaL 昇腾分支

这一步在容器内拉取 AReaL 仓库并安装 `ascend` 分支，作用是拿到已经适配昇腾平台的版本。

```bash
docker exec -it areal_npu /bin/bash
git clone https://github.com/inclusionAI/AReaL
cd AReaL
git checkout ascend
pip install -e .
```
![image.png](https://raw.gitcode.com/user-images/assets/9392763/30d6b0a4-1594-4774-8b34-dfef61a87d80/image.png 'image.png')
## 5. 检查并调整示例配置

训练脚本：

`examples/math/gsm8k_rl.py`

配置文件：

`examples/math/gsm8k_grpo_npu.yaml`

修改配置文件gsm8k_grpo_npu.yaml将模型配置为Qwen3-0.6B模型：

![image.png](https://raw.gitcode.com/user-images/assets/9392763/6241f715-e3db-4f68-ab52-f527ab64ed9c/image.png 'image.png')

修改配置文件gsm8k_grpo_npu.yaml调整训推的卡资源分配以及并行方式，默认为4卡推理+4卡训练，都使用DP并行，下面给出调整为单卡推理+单卡训练的配置调整方式： 
![image.png](https://raw.gitcode.com/user-images/assets/9392763/94373232-3d93-49d5-b745-79ecce6efdbc/image.png 'image.png')

## 6. 启动 RL 训练
训练过程会访问huggingface下载模型和数据集,若因网络原因无法访问huggingface导致模型或数据集下载失败可第七节视频处理.
```bash
python -m areal.launcher.local examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo_npu.yaml
```


当图中信息循环显示时RL训练便在正常运行了：
![image.png](https://raw.gitcode.com/user-images/assets/9392763/5560cbc0-6ef9-411b-b847-dd6cc0b00fbb/image.png 'image.png')


训练完成显示如下：
![image.png](https://raw.gitcode.com/user-images/assets/9392763/61875cbd-15b8-43ac-980a-002825b0c031/image.png 'image.png')


训练结束后新的模型文件默认在`/tmp/areal/experiments/`下，可通过`gsm8k_grpo_npu.yaml`配置文件`fileroot`参数调整文件路径:
![image.png](https://raw.gitcode.com/user-images/assets/9392763/de9b256b-936b-4bd0-9cf2-c75cd9303f98/image.png 'image.png')


## 7.参考视频
https://www.bilibili.com/video/BV1thc6z7E4U/?spm_id_from=333.337.search-card.all.click
