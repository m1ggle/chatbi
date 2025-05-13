# ChatBI

ChatBI 是一个基于自然语言处理和向量数据库的智能对话系统，旨在为用户提供高效、智能的问答和数据分析服务。

## 功能特性

- **自然语言处理**：支持多种语言模型，包括 Qwen2.5 和 OpenAI GPT。
- **向量数据库**：集成 Milvus 向量数据库，支持高效的向量存储和检索。
- **自动化提交**：提供自动化 Git 提交脚本，简化开发流程。
- **多模型支持**：支持多种 Embedding 模型，灵活适配不同场景。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境

在 `config.yaml` 中配置 Milvus 和 OpenAI 的 API 密钥：

```yaml
milvus:
  host: "localhost"
  port: 19530
  lite_mode: true

openai:
  api_key: "your_openai_api_key"
```

### 运行示例

```bash
python main.py
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！请确保代码风格一致，并附上详细的描述。

## 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。