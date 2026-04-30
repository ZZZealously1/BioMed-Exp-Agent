# BioMed-Exp Agent 文档

## 概述

BioMed-Exp Agent 4.0-R 是一个基于开源生态构建的生物医学实验智能体系统。

## 文档目录

- [安装指南](installation.md)
- [快速开始](quickstart.md)
- [API 参考](api_reference.md)

## 核心特性

- **自然语言驱动**: 一句话完成实验分析配置
- **自主修复**: 自动检测质量问题并修复
- **科学严谨**: 完整的实验谱系记录
- **经验复用**: 新实验类型快速配置

## 系统架构

```
用户请求 → Agent Core → MCP Tools → 科学约束 → 结果输出
                ↓
            记忆层
```

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | LangGraph |
| LLM 路由 | LiteLLM |
| 记忆层 | Mem0 |
| 向量数据库 | ChromaDB |
| 工具协议 | MCP |
| API 框架 | FastAPI |
