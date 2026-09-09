# Git 基础课程

本课程是版本控制工具 Git 的入门通识课，面向零基础初学者，以 Jupyter Notebook 形式提供，支持在线交互式运行。课程从版本控制的基本概念讲起，依次覆盖本地操作、分支管理、远程协作，并配有随时可查的指令速查表，最终落脚于**在 GitCode 平台提交 Pull Request 的完整流程**（从 Fork 仓库开始，涵盖 Linux 命令行与网页两种方式）。

## 课程内容

| 序号 | 课程 | 内容概要 |
|:----:|------|---------|
| 1 | [Git 简介与版本控制基础](./01_git_intro.ipynb) | 版本控制概念 → Git 诞生与核心特点 → 三个区域（办公桌/文件篮/档案柜） → Linux 安装与首次配置 |
| 2 | [Git 本地基本操作](./02_git_local.ipynb) | 仓库准备 → add/status → commit（Conventional Commits、Co-authored-by） → log → diff → 撤销操作 → .gitignore |
| 3 | [分支管理](./03_git_branch.ipynb) | 分支本质（核心原理 + 书签三步图解） → 创建/切换/删除 → 分支模型与最佳实践 |
| 4 | [远程仓库协作](./04_git_remote.ipynb) | 远程仓库概念 → HTTPS/SSH 克隆 → remote 管理 → push/pull/fetch → 跟踪分支 → 同步 upstream |
| 5 | [Git 指令汇总](./05_git_order.ipynb) | 按场景分类的指令速查表：配置初始化 / 三区流转 / 差异历史 / 撤销回滚 / Stash / 分支协作 / Tag / 远程同步 |
| 6 | [🌟 GitCode 平台 Fork 与 PR 完整流程](./06_gitcode_fork_pr.ipynb) | Fork 概念 → 网页 Fork → Linux clone/配置 upstream/创建分支 → 修改提交推送 → 网页发起 PR → 应对 Review → 合并清理 → 纯网页方式 → 完整速查表 |

> 💡 **第 6 课是本课程的重头戏**，完整演示了在 GitCode 平台上从 Fork 到 PR 合并的全流程，同时覆盖 **Linux 命令行**和**纯网页**两条路径。

## 核心内容速览

学完本课程，你应该牢固掌握以下内容。

### 三个区域与日常循环（第 1-2 课）

Git 把项目分成三个区域，用「档案管理」类比：**工作区**（办公桌）→ **暂存区**（待归档文件篮）→ **仓库**（档案柜）。日常循环就是：

```bash
git status              # 看看改了什么
git add <file>          # 把改动放进文件篮
git commit -m "msg"     # 登记入柜（msg 遵循 Conventional Commits 规范）
git log --oneline       # 查看历史
```

### 分支只是一个名字（第 3 课）

分支不复制代码，它只是指向某次提交的名字（书签）；提交只推进你所在的分支。所以开分支近乎免费、删分支不伤代码：

```bash
git checkout -b <name>  # 创建并切换到新分支（最常用）
git branch -d <name>    # 删除分支
```

### 远程与双远程模型（第 4 课）

`clone` 得到本地仓库，远程默认叫 **origin**。Fork 工作流需要双远程：

```bash
git remote -v                   # origin = 你的 fork（可 push）
git remote add upstream <url>   # upstream = 原仓库（只 fetch）
git push -u origin <branch>     # 推到自己的 fork
```

### Fork + PR 六步（第 6 课）

贡献开源的完整旅程：**Fork（网页）→ clone 自己的 fork → 建分支改代码 → push 到 fork → 发起 Pull Request → 应对 Review 直至合入**。关键心法：

1. **origin = 你的 fork**（push 目标），**upstream = 原仓库**（只能 PR 请求合入）
2. **永远在功能分支上开发**，不直接改 master
3. **PR 与分支绑定**：按 Review 意见修改后 push 同一分支，PR 自动更新，不要关闭重开
4. 需要更多指令时查 [第 5 课速查表](./05_git_order.ipynb)

## 适用人群

- 零基础初学者，想系统学习 Git 版本控制
- 有一定 Git 经验，想了解 GitCode 平台 Fork + PR 工作流的开发者
- 准备向开源项目（如 cann-learning-hub）贡献代码的贡献者

## 学习建议

1. 按顺序学习：概念（第 1 课）→ 本地操作（第 2-3 课）→ 远程协作（第 4 课）→ PR 实战（第 6 课）
2. 第 2-4、6 课穿插可执行代码 cell，建议在 Notebook 环境中逐个运行体验
3. 第 5 课是指令速查表，不必通读——收藏它，用到时查阅
4. 第 6 课是重头戏，建议结合实际的 GitCode 仓库完整走一遍
5. 学完后，推荐向 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) 提交一个练手 PR（如修复文档错别字）

## 目录结构

```text
git_basics/
├── README.md                       # 课程说明
├── 01_git_intro.ipynb              # 第 1 课：Git 简介与版本控制基础
├── 02_git_local.ipynb              # 第 2 课：Git 本地基本操作
├── 03_git_branch.ipynb             # 第 3 课：分支管理
├── 04_git_remote.ipynb             # 第 4 课：远程仓库协作
├── 05_git_order.ipynb              # 第 5 课：Git 指令汇总
├── 06_gitcode_fork_pr.ipynb        # 第 6 课：GitCode 平台 Fork 与 PR 完整流程
└── ...
```
