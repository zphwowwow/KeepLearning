# api.py
import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'E:/enviroment/HuggingFaceCache'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 添加项目根目录到 sys.path（假设 api.py 位于项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.retrieval.vector_store import ChromaStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="BIOS RAG API")

# 允许跨域（Gradio 前端需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化向量库和 LLM
store = ChromaStore(persist_directory="data/vector_store")
llm = ChatOllama(model="qwen2.5:7b")  # 请确保已拉取该模型

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的 BIOS 技术问答助手。请仅根据以下提供的参考信息回答问题。"
               "如果信息不足以回答，请坦诚告知，不要编造答案。回答请使用中文。"),
    ("user", "参考信息：\n{context}\n\n问题：{question}")
])


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class Source(BaseModel):
    source_id: str
    page: int
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # 1. 检索
    results = store.search(request.question, top_k=request.top_k, collection_name="bios_kb")
    if not results:
        context = "未找到相关文档。"
        sources = []
    else:
        context_parts = []
        sources = []
        for r in results:
            meta = r['metadata']
            src = Source(
                source_id=meta.get('source_id', '未知'),
                page=meta.get('page', -1),
                content_preview=r['text'][:150]
            )
            sources.append(src)
            context_parts.append(f"[来源：{src.source_id} 第{src.page}页]\n{r['text']}")
        context = "\n\n".join(context_parts)

    # 2. 生成回答
    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": request.question})
    return QueryResponse(answer=response.content, sources=sources)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)