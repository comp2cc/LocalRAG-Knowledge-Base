import streamlit as st
import uuid
import tempfile
import os
from src.ingestion import load_documents, chunk_documents
from src.embedding import get_chroma_collection, add_documents_to_chroma
from src.retrieval import HybridRetriever, Reranker
from src.generation import LLMGenerator

st.set_page_config(page_title="本地知识库问答", layout="wide")
st.title("📚 LocalRAG：私有知识库智能问答")

# 初始化组件
@st.cache_resource
def init_components():
    collection = get_chroma_collection()
    retriever = HybridRetriever(collection)
    reranker = Reranker()
    llm = LLMGenerator()
    return collection, retriever, reranker, llm

collection, retriever, reranker, llm = init_components()

# 会话状态
if 'current_chunks' not in st.session_state:
    st.session_state.current_chunks = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # 存储 (user, assistant)
if 'messages' not in st.session_state:
    st.session_state.messages = []      # 用于显示在聊天界面

# 侧边栏：文档管理
with st.sidebar:
    st.header("📂 知识库管理")
    uploaded_files = st.file_uploader("上传文档（PDF/TXT/MD/DOCX/HTML/CSV）", 
                                      accept_multiple_files=True,
                                      type=["pdf","txt","md","docx","html","csv"])
    if st.button("建立/更新索引") and uploaded_files:
        with st.spinner("处理文档中..."):
            temp_paths = []
            for file in uploaded_files:
                ext = file.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(file.getbuffer())
                    temp_paths.append(tmp.name)
            docs = load_documents(temp_paths)
            if not docs:
                st.error("没有可用的文档内容，请检查文件格式。")
                st.stop()
            chunks = chunk_documents(docs)
            ids = [str(uuid.uuid4()) for _ in chunks]
            add_documents_to_chroma(collection, chunks, ids)
            st.session_state.current_chunks = chunks
            retriever.build_bm25(chunks)
            for p in temp_paths:
                os.unlink(p)
            st.success(f"已索引 {len(chunks)} 个文本片段")
        st.rerun()
    
    if st.button("清空对话历史"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("请输入你的问题"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 检索
    if not st.session_state.current_chunks:
        st.warning("请先上传文档并建立索引")
        st.stop()
    
    with st.spinner("检索中..."):
        candidates = retriever.hybrid_search(prompt, top_k=10, alpha=0.6)
        if not candidates:
            answer = "未检索到相关内容，请尝试换个问题或上传相关文档。"
        else:
            reranked = reranker.rerank(prompt, candidates, top_k=5)
            with st.spinner("生成答案中..."):
                answer = llm.generate(prompt, reranked, history=st.session_state.chat_history)
    
    # 显示助手回答
    with st.chat_message("assistant"):
        st.markdown(answer)
        # 如果有检索到的片段，可以折叠显示
        if candidates:
            with st.expander("📖 引用来源"):
                for i, ctx in enumerate(reranked):
                    st.text(f"片段 {i+1}: {ctx[:300]}..." if len(ctx)>300 else ctx)
    
    # 保存历史
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append((prompt, answer))