# 第5章 PD 分离与 KV Cache 传输实战

## 章节概述
本章讲解大模型推理场景下的 Prefill/Decode（PD）分离架构，包括 PD 分离技术原理、HIXL llm-datadist 接口体系及在 omni-infer 框架中的应用，并通过实战练习掌握 P→D 的 KV Cache 传输方法。

## 在线体验
| Notebook | Link | 状态 |
|--|--|--|
| 5.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/05_pd_separation_kv_cache_transfer/05.01_chapter_intro.ipynb&npuCnt=2) | 🔄 待发布 |
| 5.2 PD 分离技术概述 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/05_pd_separation_kv_cache_transfer/05.02_pd_separation_overview.ipynb&npuCnt=2) | 🔄 待发布 |
| 5.3 HIXL llm-datadist 接口详解 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/05_pd_separation_kv_cache_transfer/05.03_llm_datadist_api.ipynb&npuCnt=2) | 🔄 待发布 |
| 5.4 Mooncake TE 概述 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/05_pd_separation_kv_cache_transfer/05.04_mooncake_transfer_engine_overview.ipynb&npuCnt=2) | 🔄 待发布 |
| 5.5 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=ubuntu22-cann9.0-python3.11-jupyter%3Av1.0.4-hixl-notebook&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/05_pd_separation_kv_cache_transfer/05.05_chapter_practice.ipynb&npuCnt=2) | 🔄 待发布 |
