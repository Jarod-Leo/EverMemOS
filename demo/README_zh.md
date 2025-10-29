# Demo - EverMemOS 交互式示例

[English](README.md) | [简体中文](README_zh.md)

本目录包含展示 EverMemOS 核心功能的交互式演示。

## 📂 内容

### 核心演示脚本

- **`extract_memory.py`** - 从对话数据中提取记忆
  - 处理 `data/` 目录中的对话文件
  - 提取记忆单元（MemCells）并生成用户画像
  - 将结果保存到配置的数据库（MongoDB）和本地输出

- **`chat_with_memory.py`** - 与记忆增强 AI 进行交互式对话
  - 用于与 AI 智能体对话的命令行界面
  - 利用提取的记忆进行上下文感知的回应
  - 演示端到端的记忆检索和使用

### 配置文件

- **`memory_config.py`** - 记忆系统配置
- **`memory_utils.py`** - 记忆操作的工具函数

### 输出目录

- **`chat_history/`** - 保存的聊天对话记录（自动生成）
- **`memcell_outputs/`** - 提取的 MemCell 输出（自动生成）

## 🚀 快速开始

### 1. 提取记忆

首先，从示例对话数据中提取记忆：

```bash
uv run python src/bootstrap.py demo/extract_memory.py
```

这将：
- 从 `data/` 目录读取对话数据
- 使用 LLM 提取记忆单元
- 生成用户画像
- 存储到 MongoDB 并保存到本地文件

### 2. 与记忆对话

提取完成后，启动交互式聊天会话：

```bash
uv run python src/bootstrap.py demo/chat_with_memory.py
```

功能特性：
- 与 AI 实时对话
- 基于上下文自动检索记忆
- 双语支持（英语/中文）
- 聊天历史自动保存

## 📝 使用提示

### 记忆提取

提取脚本处理 `data/` 目录中的 JSON 文件（在 `memory_config.py` 的 `ExtractModeConfig` 中配置）。请确保您的对话文件遵循 [GroupChatFormat](../data_format/group_chat/group_chat_format.md) 规范。为了方便，我们为您提供了双语示例数据文件：
- **英文版本：**[group_chat_en.json](../data/group_chat_en.json) 和 [assistant_chat_en.json](../data/assistant_chat_en.json)
- **中文版本：**[group_chat_zh.json](../data/group_chat_zh.json) 和 [assistant_chat_zh.json](../data/assistant_chat_zh.json)

详情请查看[数据说明文档](../data/README_zh.md)。

### 交互式聊天

在聊天会话期间：
- 自然地输入消息与 AI 对话
- 系统自动检索相关记忆
- 使用 `exit` 结束会话
- 聊天历史自动保存（默认读取前 5 条上下文）

## 🔧 配置

编辑 `memory_config.py` 以自定义：
- LLM 模型选择
- 记忆提取参数
- 数据库连接设置
- 输出目录

## 📊 示例输出

运行 `extract_memory.py` 后，您将找到：

```
demo/memcell_outputs/
├── user_profiles/
│   ├── user_101_profile.json
│   └── user_102_profile.json
└── memcells/
    ├── group_001_memcells.json
    └── episode_memories.json
```

## 🔗 相关文档

- [群聊格式规范](../data_format/group_chat/group_chat_format.md)
- [API 文档](../docs/api_docs/agentic_v3_api_zh.md)
- [数据说明文档](../data/README_zh.md)
- [国际化使用指南](../docs/dev_docs/chat_i18n_usage.md)

## 📖 演示数据说明

### 群聊场景 (group_chat_en.json / group_chat_zh.json)

**项目背景：** AI 产品工作群，记录团队开发"智能销售助手"的完整历程

**核心内容：**
- MVP 开发阶段：基于 RAG 的问答系统
- 高级功能迭代：情绪识别、记忆系统
- 团队协作实践：从需求到交付的完整流程

**可用语言：** 英文和中文版本

**适合探索：** 团队协作模式、项目管理、技术方案演进

### 助手场景 (assistant_chat_en.json / assistant_chat_zh.json)

**对话背景：** 个人健康与生活助手，记录近 2 个月的连续交互

**核心内容：**
- 旅行规划：美食推荐、行程建议
- 健康管理：体重监测、饮食指导
- 运动康复：训练建议、伤后恢复

**可用语言：** 英文和中文版本

**适合探索：** 个性化服务、长期记忆积累、上下文理解

## ❓ 推荐问题示例

**群聊 AI 场景问题推荐：**
- Alex/Betty/... 在情绪识别项目中做了什么工作？
- 从情绪识别项目中，可以看出 Alex/Betty/... 具备什么样的工作能力？
- 情绪识别项目的交付结果如何？
- 记忆系统项目的进展如何？

**助手 AI 场景问题推荐：**
- 请为我推荐适合我的运动。
- 请为我推荐我可能喜欢的食物。
- 我的健康状况如何？

## 💡 需要帮助？

- 查看主 [README](../README_zh.md) 了解设置说明
- 查阅 [批量记忆化使用指南](../docs/dev_docs/run_memorize_usage.md)
- 在 GitHub 上提交问题

---

**祝您探索愉快！🧠✨**

