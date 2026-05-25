import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import streamlit as st
import uuid
import tempfile
import os
from src.ingestion import load_documents, chunk_documents
from src.embedding import get_chroma_collection, add_documents_to_chroma
from src.retrieval import HybridRetriever, Reranker
from src.generation import LLMGenerator
from src.agent.safety_agent import run_agent   # 新增：导入 Agent

st.set_page_config(page_title="本地知识库问答", layout="wide")
st.title("📚 LocalRAG：私有知识库智能问答")

# 初始化组件（缓存）
@st.cache_resource
def init_components():
    collection = get_chroma_collection()
    retriever = HybridRetriever(collection)
    reranker = Reranker()
    llm = LLMGenerator()
    return collection, retriever, reranker, llm

collection, retriever, reranker, llm = init_components()

# 会话状态初始化
if 'current_chunks' not in st.session_state:
    st.session_state.current_chunks = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []   # 存储 (user, assistant) 用于多轮对话记忆
if 'messages' not in st.session_state:
    st.session_state.messages = []       # 用于显示聊天历史
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())   # 匿名用户标识，用于黑名单

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("📂 知识库管理")
    uploaded_files = st.file_uploader(
        "上传文档（PDF/TXT/MD/DOCX/HTML/CSV）",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "docx", "html", "csv"]
    )
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

    st.divider()
    # 清空对话历史按钮
    if st.button("清空对话历史"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    # Agent 安全审核开关
    enable_agent = st.checkbox("🛡️ 启用安全审核 Agent", value=False)
    if enable_agent:
        st.info("Agent 将检查您的输入是否违规，违规内容将被拦截。")
        st.caption(f"当前用户ID: {st.session_state.user_id[:8]}...")

# ==================== 主界面（聊天历史）====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================== 用户输入处理 ====================
if prompt := st.chat_input("请输入你的问题"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ----- Agent 审核（如果启用）-----
    if enable_agent:
        with st.spinner("安全审核中..."):
            final_decision, reason = run_agent(prompt, st.session_state.user_id)
        if "违规" in final_decision:
            # 拦截：不进行 RAG 回答
            with st.chat_message("assistant"):
                st.error(f"🚫 内容被拦截：{final_decision}")
                st.caption(f"原因：{reason}")
            st.session_state.messages.append({"role": "assistant", "content": f"内容被拦截：{final_decision}"})
            st.stop()   # 停止后续处理
        else:
            # 审核通过，可选显示成功提示
            with st.chat_message("assistant"):
                st.success("✅ 安全审核通过")
            # 注意：这里显示的提示会作为一条独立消息，但不在历史中保存（可保持简洁）
            # 实际不把这条提示加入 messages 历史，以免干扰对话。也可以不加提示，直接继续。
            # 为了体验，删除刚添加的成功消息（如果不希望显示额外气泡，可以注释掉）
            # 简单的做法是不显示额外消息，直接继续。下面直接继续 RAG 流程。

    # ----- 原有 RAG 问答流程 -----
    if not st.session_state.current_chunks:
        # 未上传文档时的兜底提示
        answer = "⚠️ 请先上传文档并建立索引。"
        with st.chat_message("assistant"):
            st.warning(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        with st.spinner("检索中..."):
            candidates = retriever.hybrid_search(prompt, top_k=10, alpha=0.6)
            if not candidates:
                answer = "未检索到相关内容，请尝试换个问题或上传相关文档。"
            else:
                reranked = reranker.rerank(prompt, candidates, top_k=5)
                with st.spinner("生成答案中..."):
                    # 使用多轮对话记忆（最近3轮）
                    history = st.session_state.chat_history[-6:]
                    answer = llm.generate(prompt, reranked, history=history)
        # 显示助手回答
        with st.chat_message("assistant"):
            st.markdown(answer)
            # 如果有检索到的片段，折叠显示引用来源
            if candidates:
                with st.expander("📖 引用来源"):
                    for i, ctx in enumerate(reranked):
                        preview = ctx[:300] + "..." if len(ctx) > 300 else ctx
                        st.text(f"片段 {i+1}: {preview}")

        # 保存历史（包括对话历史和显示用消息）
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append((prompt, answer))