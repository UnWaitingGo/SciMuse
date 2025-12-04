# 实现SciMuse的RAG流程计划

## 1. 项目结构搭建

* 创建Readme.md中定义的目录结构：agents、models、tools、data、experiments、docs

* 创建requirements.txt文件，添加必要的依赖

* 创建config.yaml配置文件

## 2. 环境配置

* 生成.env文件，存储modelscope访问令牌

* 配置OpenAI兼容的客户端，用于调用modelscope API

## 3. 核心组件实现

### 3.1 工具函数

* **pdf\_parser.py**：使用MinerU实现PDF解析，提取文字、图片、公式

* **vector\_db.py**：实现向量数据库功能，基于ChromaDB

### 3.2 Agent实现

* **Planner Agent**：任务分解，将用户问题拆分为子任务

* **Retriever Agent**：多模态检索，使用Qwen3-Embedding和向量数据库

* **Caption Agent**：图像理解，使用Qwen3-VL+MinerU

* **Reasoner Agent**：推理生成，基于VL系列模型

* **Reviewer Agent**：自我校验，基于规则和LLM Judge

### 3.3 模型集成

* 实现Qwen3-Embedding调用

* 实现Qwen3-VL调用

* 实现Qwen3-Coder调用

## 4. 主程序实现

* 创建main.py，实现完整的RAG流程

* 确保各Agent之间的协作和数据流转

## 5. 测试和验证

* 确保test\_agent.py能够正常运行

* 验证PDF解析、向量存储、检索、生成和审核功能

* 确保输出带有引用和置信度

## 技术栈

* Python 3.8+

* OpenAI兼容API（调用modelscope）

* ChromaDB向量数据库

* MinerU PDF解析

* YAML配置管理

* smolagents 框架mvp实现

* Qwen3 家族模型

## 注意事项

* 所有模型调用都通过API进行，不使用本地模型

* 确保.env文件的安全性，不泄露访问令牌

* 实现模块化设计，便于扩展和维护

* 遵循用户提供的API调用示例

## 预期成果

* 完整的SciMuse RAG流程实现

* 能够处理用户问题和PDF文件

* 输出带有引用和置信度的答案

* 支持多模态检索和图像理解

- 实现 Planning agent+4个sub agent 协作进行处理任务
- 支持 PDF 多模态解析 + 图像理解
- 输出带引用、置信度的答案
- mcp

