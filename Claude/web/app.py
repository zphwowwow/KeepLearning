"""
web.app — FastAPI Web 后端

本模块是 BMC 管理助手的 Web 服务端，提供:
    1. WebSocket 长连接对话接口（/ws）
    2. 静态前端页面托管（/ 和 /static/）
    3. REST API 配置查询接口（/api/config）

架构:
    浏览器 ←WebSocket→ FastAPI ←invoke→ LangGraph ←tool_call→ BMCManager → ipmitool.exe

WebSocket 通信协议:
    客户端 → 服务器:  {"message": "查看电源状态"}
    服务器 → 客户端:
        {"type": "user", "content": "..."}          # 回显用户消息
        {"type": "tool_call", "name": "...", "args": {...}}  # 工具调用通知
        {"type": "tool_result", "name": "...", "content": "..."}  # 工具执行结果
        {"type": "assistant", "content": "..."}     # LLM 最终回复
        {"type": "error", "content": "..."}         # 错误信息

生命周期管理:
    使用 FastAPI 的 lifespan 机制管理 SQLite checkpointer 的生命周期，
    确保数据库连接在服务启动时打开、关闭时正确释放。
"""

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bmc_agent.config import load_config, create_llm
from bmc_agent.graph import build_graph
from bmc_agent.manager import BMCManager
from bmc_agent.memory import get_sqlite_checkpointer
from bmc_agent.runner import IPMIToolRunner
from bmc_agent.tools import set_manager

# ══════════════════════════════════════════════════════════════════════
# 配置加载与核心组件初始化
# ══════════════════════════════════════════════════════════════════════
# 这些组件在模块导入时初始化，整个应用生命周期共享同一实例。

config = load_config()    # 加载 config.yaml 配置
llm = create_llm(config)  # 创建 LLM 实例（硅基流动 API）

runner = IPMIToolRunner(  # 创建 IPMI 命令执行器
    host=config["bmc"]["host"],
    username=config["bmc"]["username"],
    password=config["bmc"]["password"],
    interface=config["bmc"]["interface"],
    port=config["bmc"]["port"],
    timeout=config["bmc"]["timeout"],
    retries=config["bmc"]["retries"],
)
manager = BMCManager(runner)  # 创建 BMC 操作管理器
set_manager(manager)           # 注入到 tools 模块的全局变量

# ══════════════════════════════════════════════════════════════════════
# FastAPI 应用与生命周期
# ══════════════════════════════════════════════════════════════════════

_sqlite_ctx = None  # SQLite checkpointer 的上下文管理器引用
graph = None        # LangGraph 编译后的状态图


@asynccontextmanager
async def _lifespan(app):
    """FastAPI 应用生命周期管理器。

    启动时:
        1. 打开 SQLite checkpointer（创建数据库连接）
        2. 初始化数据库表结构（checkpointer.setup()）
        3. 构建 LangGraph 状态图并注入 checkpointer

    关闭时:
        1. 自动关闭 SQLite 数据库连接（上下文管理器 __exit__）
    """
    global _sqlite_ctx, graph
    _sqlite_ctx = get_sqlite_checkpointer(config["memory"]["db_path"])
    checkpointer = _sqlite_ctx.__enter__()  # 手动进入上下文
    checkpointer.setup()                     # 初始化 SQLite 表
    graph = build_graph(llm, checkpointer=checkpointer)
    yield
    _sqlite_ctx.__exit__(None, None, None)   # 手动退出上下文，释放连接


app = FastAPI(lifespan=_lifespan)

# ══════════════════════════════════════════════════════════════════════
# 静态文件与首页
# ══════════════════════════════════════════════════════════════════════

app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.get("/")
async def index():
    """返回前端聊天页面 HTML。"""
    with open("web/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ══════════════════════════════════════════════════════════════════════
# WebSocket 对话接口
# ══════════════════════════════════════════════════════════════════════
# 每个 WebSocket 连接分配独立的 thread_id，LangGraph 的 checkpointer
# 会按 thread_id 隔离不同连接的对话历史。

@app.websocket("/ws")
async def chat(ws: WebSocket):
    """WebSocket 对话端点。

    工作流程:
        1. 接受 WebSocket 连接，分配唯一 thread_id
        2. 循环接收用户消息
        3. 调用 LangGraph graph.invoke() 执行 Agent 推理
        4. 遍历结果中的消息，将工具调用和 AI 回复推送给客户端
        5. 连接断开时退出循环

    消息格式:
        输入: {"message": "查看电源状态"}
        输出: 见模块文档头部的协议说明
    """
    await ws.accept()
    thread_id = str(uuid.uuid4())

    try:
        while True:
            raw = await ws.receive_text()
            # 解析客户端消息，支持 JSON 和纯文本两种格式
            try:
                data = json.loads(raw)
                user_msg = data.get("message", "").strip()
            except json.JSONDecodeError:
                user_msg = raw.strip()

            if not user_msg:
                continue

            # 检查 graph 是否已初始化（lifespan 可能尚未完成）
            if graph is None:
                await ws.send_json({"type": "error", "content": "服务尚未就绪"})
                continue

            # 回显用户消息（前端已本地添加，此处仅作确认）
            await ws.send_json({"type": "user", "content": user_msg})

            # 调用 LangGraph 执行 Agent 推理循环
            try:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_msg)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
            except Exception as e:
                await ws.send_json({"type": "error", "content": str(e)})
                continue

            # 遍历结果消息，推送给前端
            for msg in result["messages"]:
                if isinstance(msg, HumanMessage):
                    continue  # 跳过用户消息（已回显）
                elif isinstance(msg, AIMessage):
                    # AI 消息可能同时包含工具调用和文本内容
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            await ws.send_json({
                                "type": "tool_call",
                                "name": tc["name"],
                                "args": tc["args"],
                            })
                    if msg.content:
                        await ws.send_json({
                            "type": "assistant",
                            "content": msg.content,
                        })
                elif isinstance(msg, ToolMessage):
                    # 工具执行结果，截断过长内容避免 WebSocket 帧过大
                    await ws.send_json({
                        "type": "tool_result",
                        "name": msg.name or "",
                        "content": msg.content[:2000] if msg.content else "",
                    })

    except WebSocketDisconnect:
        pass  # 客户端断开连接，正常退出


# ══════════════════════════════════════════════════════════════════════
# REST API 接口
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/config")
async def get_config_info():
    """返回当前配置信息供前端展示（不包含密码等敏感信息）。"""
    return {
        "bmc_host": config["bmc"]["host"],
        "bmc_username": config["bmc"]["username"],
        "model": config["llm"]["model"],
    }
