# LocalRAG-Knowledge-Base

> 基于混合检索与重排序的私有知识库问答系统 | 集成内容安全审核 Agent | RAG | LangGraph | Streamlit

## ✨ 功能特点

- 📚 **多格式文档支持**：PDF, TXT, Markdown, DOCX, HTML, CSV
- 🔍 **混合检索**：向量检索 (BGE) + BM25 关键词检索，提升召回率
- 🎯 **重排序**：Cross-Encoder 模型精排检索结果
- 💬 **多轮对话记忆**：记住上下文，支持连续问答
- 🛡️ **内容安全审核 Agent**：基于 LangGraph 构建，集成敏感词检测、本地分类模型、用户黑名单管理
- 🌐 **Web 界面**：Streamlit 构建，开箱即用
- 🐳 **Docker 化**：一键部署（可选）

## 🛠️ 技术栈

- **前端**：Streamlit
- **向量数据库**：ChromaDB
- **Embedding 模型**：BAAI/bge-small-zh-v1.5
- **重排序模型**：cross-encoder/ms-marco-MiniLM-L-6-v2
- **大语言模型**：DeepSeek API（兼容 OpenAI）
- **框架**：LangChain, LangGraph
- **容器化**：Docker, docker-compose

## 🚀 快速开始

### 环境要求
- Python 3.10+
- pip

### 1. 克隆仓库
```bash
git clone https://github.com/comp2cc/LocalRAG-Knowledge-Base.git
cd LocalRAG-Knowledge-Base
