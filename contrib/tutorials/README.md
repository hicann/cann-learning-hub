# 外部贡献课程

本目录用于存放社区开发者贡献的 CANN 学习课程和教程。

## 贡献指南

### 目录结构

每个贡献的课程应放在独立文件夹中，结构示例：

```
contrib/tutorials/
├── your_course_name/
│   ├── README.md          # 课程介绍
│   ├── chapter_1/          # 章节内容
│   │   └── notebook.ipynb
│   └── chapter_2/
│       └── notebook.ipynb
└── README.md              # 本文件
```

### 提交要求

1. 每个课程需要包含 `README.md` 介绍课程内容和目标
2. 课程介绍中可以署上作者相关信息（姓名、联系方式等）
3. Notebook 文件需使用 UTF-8 编码
4. 代码中避免硬编码路径，使用相对路径或环境变量
5. 提交前请在本地验证 Notebook 可以正常执行

### 审核流程

1. Fork 本仓库
2. 在 `contrib/tutorials/` 下创建您的课程目录
3. 提交 Pull Request 到 `cann/cann-learning-hub` 的 `master` 分支
4. 等待审核通过后合并

## 贡献者列表

| 贡献者 | 课程名称 | 贡献日期 | 状态 |
|--------|----------|----------|------|
| Datawhale / Torch-RecHub 社区 | [Torch-RecHub 推荐系统实战教程](./torch-rechub) | 2026.05 | 已迁移 |

## 课程列表

| 教程名称 | 教程描述 | 访问链接 | 状态 |
|----------|----------|----------|------|
| Torch-RecHub 推荐系统实战教程 | 基于 Torch-RecHub 的推荐系统端到端实战教程，覆盖 CTR 精排、序列兴趣建模、召回、多任务学习、实验跟踪与模型导出 | [torch-rechub](./torch-rechub) | 已迁移 |

---

如果您有任何问题，欢迎提交 Issue 或联系社区维护者。
