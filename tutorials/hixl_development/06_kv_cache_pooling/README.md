# 第6章 KV Cache 池化技术介绍

## 章节概述
本章讲解大模型推理场景下的 KV Cache 池化技术，包括池化技术原理、Mooncake Store 架构设计、在昇腾 NPU 上的安装与使用方法，并通过实战练习掌握跨节点 KV Cache 共享的实现。

## 在线体验
| Notebook | Link | 状态 |
|--|--|--|
| 6.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/06_kv_cache_pooling/06.01_chapter_intro.ipynb&npuCnt=2) | 🔄 待发布 |
| 6.2 KV Cache 池化技术概述 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/06_kv_cache_pooling/06.02_kv_cache_pooling_overview.ipynb&npuCnt=2) | 🔄 待发布 |
| 6.3 Mooncake Store 介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/06_kv_cache_pooling/06.03_mooncake_store_intro.ipynb&npuCnt=2) | 🔄 待发布 |
| 6.4 Mooncake Store 在昇腾 NPU 上的使用 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/06_kv_cache_pooling/06.04_mooncake_npu_usage.ipynb&npuCnt=2) | 🔄 待发布 |
| 6.5 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/06_kv_cache_pooling/06.05_chapter_practice.ipynb&npuCnt=2) | 🔄 待发布 |
