# 第三方框架集成 npugraph_ex 实践手册

本实践目标是跑通案例中的两个核心功能：

- 用 `torch.compile` 自定义后端接入 `npugraph_ex`
- 为编译结果增加缓存，减少二次启动时的编译耗时

## 1. 环境要求

|依赖  |版本要求  |  
|--|--|
| 硬件 | A2、A3系列硬件(单卡) |
|CANN版本|8.5+|
|PyTorch|2.9.0|
|torch_npu|7.3.0(需要和CANN版本配套)|

## 2. 用自定义后端接入 npugraph_ex
`custom_backend.py`脚本在`torch.compile`自定义后端中接入`npugraph_ex`,加速简单的加法模型推理。通过以下命令执行:
```
python3 custom_backend.py
```

## 3. 编译缓存与加载
第一次执行`cache_compile.py`会缓存图编译优化的结果到当前目录,在第二次执行时加载当前目录的编译产物省略编译过程,加速程序冷启动耗时:
第一次执行:
```
python3 cache_compile.py
```
![image.png](https://raw.gitcode.com/user-images/assets/9392763/b5f8476c-7f6f-4642-9371-f7924705a8b8/image.png 'image.png')
第二次执行:
```
python3 cache_compile.py
```
![image.png](https://raw.gitcode.com/user-images/assets/9392763/8696c52d-4b3f-462e-b7cb-0c4fd5fb523d/image.png 'image.png')
## 4.参考视频
https://www.bilibili.com/video/BV1bhc6z7EV6/?spm_id_from=333.337.search-card.all.click&vd_source=cdcc889287284ded8cea5d8be0291082
