from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from app.rag import RAGRetriever
from app.llm import LLMInference
from app.config import Config

app = FastAPI(title="BIOS QA System")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
retriever = RAGRetriever()
llm = LLMInference()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = Config.TOP_K
    include_context: Optional[bool] = False

class ContextItem(BaseModel):
    content: str
    source: str
    page: int
    score: float

class QueryResponse(BaseModel):
    answer: str
    contexts: List[ContextItem] = []

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # 检索
        contexts = retriever.retrieve(request.query, request.top_k)
        # 生成
        answer = llm.generate(request.query, contexts)
        response = QueryResponse(answer=answer)
        if request.include_context:
            response.contexts = [ContextItem(**c) for c in contexts]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)