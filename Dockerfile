FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（用于PDF和文档处理）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 下载模型缓存（可选，避免运行时下载）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]