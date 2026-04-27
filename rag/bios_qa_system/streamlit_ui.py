# chat_ui.py
import streamlit as st
import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'E:/enviroment/HuggingFaceCache'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.vector_store import ChromaStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ========== 页面配置 ==========
st.set_page_config(page_title="BIOS 知识库问答", page_icon="🔧", layout="wide")
st.title("🔧 BIOS 技术问答助手")
st.markdown("基于本地知识库 + Ollama 本地模型")

# ========== 初始化会话状态 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []  # 存储 {"role": "user/assistant", "content": ..., "sources": ...}
if "store" not in st.session_state:
    st.session_state.store = ChromaStore(persist_directory="data/vector_store")

# ========== 侧边栏配置 ==========
with st.sidebar:
    st.header("⚙️ 配置")
    top_k = st.slider("检索文档块数量 (Top‑K)", min_value=1, max_value=10, value=3,
                      help="越多则上下文越丰富，但可能引入噪声")
    temperature = st.slider("生成温度", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="越高越随机")
    model_name = st.selectbox("Ollama 模型", ["qwen2.5:7b", "llama3.2:3b", "deepseek-coder:6.7b"], index=0)

    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📚 知识库状态")
    st.info(f"向量库路径: `data/vector_store`\n集合名: `bios_kb`")


# ========== 初始化 LLM ==========
@st.cache_resource
def get_llm(model, temp):
    return ChatOllama(model=model, temperature=temp)


llm = get_llm(model_name, temperature)

# ========== 提示词模板 ==========
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的 BIOS 技术问答助手。请**仅根据以下提供的参考信息**回答问题。"
               "如果参考信息不足以回答，请坦诚告知用户，不要编造答案。回答请使用中文。"),
    ("user", "参考信息：\n{context}\n\n用户问题：{question}")
])


# ========== 辅助函数：执行 RAG ==========
def ask_question(question: str):
    # 1. 检索
    results = st.session_state.store.search(question, top_k=top_k, collection_name="bios_kb")
    if not results:
        context = "未找到相关文档。"
        sources = []
    else:
        context_parts = []
        sources = []
        for r in results:
            meta = r['metadata']
            source_id = meta.get('source_id', '未知文件')
            page = meta.get('page', '?')
            sources.append({
                "source_id": source_id,
                "page": page,
                "content_preview": r['text'][:200]
            })
            context_parts.append(f"[来源：{source_id} 第{page}页]\n{r['text']}")
        context = "\n\n".join(context_parts)

    # 2. 调用 LLM
    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content, sources


# ========== 显示历史消息 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 参考来源"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source_id']}** (第 {src['page']} 页)")
                    st.caption(src['content_preview'])

# ========== 输入框 ==========
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            answer, sources = ask_question(prompt)
        st.markdown(answer)
        if sources:
            with st.expander("📖 参考来源"):
                for src in sources:
                    st.markdown(f"**{src['source_id']}** (第 {src['page']} 页)")
                    st.caption(src['content_preview'])

    # 保存回答（含来源）
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })