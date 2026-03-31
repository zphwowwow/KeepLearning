from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Message(BaseModel):
    role: str  # user, assistant, system
    content: str

class SessionData(BaseModel):
    session_id: str
    history: List[Message] = []
    context: Dict[str, Any] = {}

class ToolCall(BaseModel):
    tool: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None