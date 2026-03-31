from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import uvicorn
from mcp_server.config import Config
from mcp_server.tools import TOOLS, execute_tool

app = FastAPI(title="MCP BIOS Tools Server")

class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCallResponse(BaseModel):
    result: Any

class ToolListResponse(BaseModel):
    tools: List[Dict]

@app.post("/tools/list", response_model=ToolListResponse)
async def list_tools():
    """返回所有可用工具的描述（符合MCP规范）"""
    tools_info = []
    for t in TOOLS:
        tools_info.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        })
    return ToolListResponse(tools=tools_info)

@app.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest):
    """调用指定工具并返回结果"""
    result = execute_tool(request.name, request.arguments)
    return ToolCallResponse(result=result)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host=Config.MCP_SERVER_HOST, port=Config.MCP_SERVER_PORT)