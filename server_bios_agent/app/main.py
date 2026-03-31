from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import run_agent

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        answer = run_agent(request.session_id, request.message)
        return ChatResponse(session_id=request.session_id, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}