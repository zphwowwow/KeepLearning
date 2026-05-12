"""
web.app — FastAPI Web 后端

本模块是 BMC 管理助手的 Web 服务端，提供:
    1. WebSocket 长连接对话接口（/ws）
    2. 静态前端页面托管（/ 和 /static/）
    3. REST API 配置查询接口（/api/config）

架构:
    浏览器 ←WebSocket→ FastAPI ←invoke→ LangGraph ←tool_call→ BMCManager → ipmitool.exe

WebSocket 通信协议:
    客户端 → 服务器:
        {"message": "查看电源状态"}                                # 用户消息
        {"type": "confirm_result", "approved": true, "tool_call_id": "..."}  # 确认结果

    服务器 → 客户端:
        {"type": "user", "content": "..."}                        # 回显用户消息
        {"type": "tool_call", "name": "...", "args": {...}}       # 工具调用通知
        {"type": "tool_result", "name": "...", "content": "..."}  # 工具执行结果
        {"type": "assistant", "content": "..."}                   # LLM 最终回复
        {"type": "confirm", "tool": "...", "message": "...", "tool_call_id": "..."}  # 危险操作确认请求
        {"type": "error", "content": "..."}                       # 错误信息

危险操作确认流程:
    当 LLM 调用的工具属于 DANGEROUS_TOOLS 时:
        1. 后端拦截 tool_call，向客户端发送 confirm 请求
        2. 前端弹出确认对话框
        3. 用户确认 → 后端执行工具
        4. 用户拒绝 → 后端发送取消消息，工具不执行

生命周期管理:
    使用 FastAPI 的 lifespan 机制管理记忆后端的生命周期。
"""

import asyncio
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
from bmc_agent.memory import get_checkpointer
from bmc_agent.runner import IPMIToolRunner
from bmc_agent.tools import set_manager

# ══════════════════════════════════════════════════════════════════════
# 危险操作定义
# ══════════════════════════════════════════════════════════════════════
# 这些工具在执行前需要前端弹窗确认，防止误操作导致服务器中断。

DANGEROUS_TOOLS = {"power_off", "power_reset", "power_cycle", "sel_clear"}
DANGEROUS_TOOLS_DESC = {
    "power_off": "远程硬关机（直接断电）",
    "power_reset": "硬重置服务器（等效按复位按钮）",
    "power_cycle": "电源循环（先断电再上电）",
    "sel_clear": "清除系统事件日志（不可恢复）",
}

# ══════════════════════════════════════════════════════════════════════
# 配置加载与核心组件初始化
# ══════════════════════════════════════════════════════════════════════

config = load_config()
llm = create_llm(config)

runner = IPMIToolRunner(
    host=config["bmc"]["host"],
    username=config["bmc"]["username"],
    password=config["bmc"]["password"],
    interface=config["bmc"]["interface"],
    port=config["bmc"]["port"],
    timeout=config["bmc"]["timeout"],
    retries=config["bmc"]["retries"],
)
manager = BMCManager(runner)
set_manager(manager)

# ══════════════════════════════════════════════════════════════════════
# FastAPI 应用与生命周期
# ══════════════════════════════════════════════════════════════════════

_checkpointer_ctx = None
graph = None
confirm_required = True  # 是否启用危险操作前端确认


@asynccontextmanager
async def _lifespan(app):
    """FastAPI 生命周期管理器。

    启动时根据 config 中 memory.backend 选择记忆后端（SQLite/Redis），
    Redis 连接失败时自动降级到 SQLite。
    """
    global _checkpointer_ctx, graph, confirm_required

    confirm_required = config.get("danger", {}).get("confirm_required", True)

    # 使用 get_checkpointer 自动选择后端（含 Redis 降级）
    checkpointer_result = get_checkpointer(config)

    # 统一处理: 上下文管理器需要 enter/exit，普通实例直接用
    checkpointer = None
    if hasattr(checkpointer_result, "__enter__"):
        # SQLite: 上下文管理器模式
        _checkpointer_ctx = checkpointer_result
        checkpointer = _checkpointer_ctx.__enter__()
        checkpointer.setup()
    else:
        # Redis: 直接使用实例
        checkpointer = checkpointer_result

    graph = build_graph(
        llm,
        checkpointer=checkpointer,
        context_cfg=config.get("context", {}),
    )
    yield

    if _checkpointer_ctx is not None:
        _checkpointer_ctx.__exit__(None, None, None)


app = FastAPI(lifespan=_lifespan)

# ══════════════════════════════════════════════════════════════════════
# 静态文件与首页
# ══════════════════════════════════════════════════════════════════════

app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.get("/")
async def index():
    with open("web/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ══════════════════════════════════════════════════════════════════════
# WebSocket 对话接口
# ══════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def chat(ws: WebSocket):
    """WebSocket 对话端点，含危险操作确认流程。"""
    await ws.accept()
    thread_id = str(uuid.uuid4())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"message": raw.strip()}

            # ── 处理确认结果 ──────────────────────────────────────
            if data.get("type") == "confirm_result":
                # 由 _process_with_confirm 内部的 pending_confirm 处理
                continue

            # ── 处理普通用户消息 ─────────────────────────────────
            user_msg = data.get("message", "").strip()
            if not user_msg:
                continue

            if graph is None:
                await ws.send_json({"type": "error", "content": "服务尚未就绪"})
                continue

            await ws.send_json({"type": "user", "content": user_msg})

            try:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_msg)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
            except Exception as e:
                await ws.send_json({"type": "error", "content": str(e)})
                continue

            # 遍历结果消息，处理危险操作确认
            await _send_messages_with_confirm(ws, result["messages"])

    except WebSocketDisconnect:
        pass


async def _send_messages_with_confirm(ws: WebSocket, messages: list):
    """遍历结果消息，对危险工具调用进行前端确认拦截。

    流程:
        1. AI 消息含 tool_calls → 检查是否危险工具
        2. 危险工具 → 发送 confirm 请求 → 等待用户确认
        3. 用户确认 → 发送 tool_call + 执行结果
        4. 用户拒绝 → 发送取消消息
        5. 非危险工具 → 直接发送 tool_call + 执行结果
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            continue

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc["name"]

                    # ── 检查是否危险工具 ─────────────────────────
                    if confirm_required and tool_name in DANGEROUS_TOOLS:
                        desc = DANGEROUS_TOOLS_DESC.get(tool_name, tool_name)
                        tool_call_id = tc.get("id", str(uuid.uuid4()))

                        # 发送确认请求到前端
                        await ws.send_json({
                            "type": "confirm",
                            "tool": tool_name,
                            "message": f"⚠ 即将执行危险操作：{desc}，确认执行吗？",
                            "tool_call_id": tool_call_id,
                        })

                        # 等待前端确认结果
                        approved = await _wait_for_confirm(ws, tool_call_id)

                        if approved:
                            # 用户确认: 显示工具调用信息
                            await ws.send_json({
                                "type": "tool_call",
                                "name": tool_name,
                                "args": tc["args"],
                            })
                        else:
                            # 用户拒绝: 通知前端
                            await ws.send_json({
                                "type": "assistant",
                                "content": f"已取消操作：{desc}",
                            })
                            continue
                    else:
                        # 非危险工具: 直接通知
                        await ws.send_json({
                            "type": "tool_call",
                            "name": tool_name,
                            "args": tc["args"],
                        })

            if msg.content:
                await ws.send_json({"type": "assistant", "content": msg.content})

        elif isinstance(msg, ToolMessage):
            await ws.send_json({
                "type": "tool_result",
                "name": msg.name or "",
                "content": msg.content[:2000] if msg.content else "",
            })


async def _wait_for_confirm(ws: WebSocket, tool_call_id: str, timeout: float = 120) -> bool:
    """等待前端确认结果。

    超时或用户拒绝返回 False，用户确认返回 True。

    Args:
        ws: WebSocket 连接
        tool_call_id: 关联的工具调用 ID
        timeout: 最长等待时间（秒），超时视为拒绝

    Returns:
        bool: 用户是否确认执行
    """
    try:
        while True:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=timeout)
            data = json.loads(raw)
            if data.get("type") == "confirm_result" and data.get("tool_call_id") == tool_call_id:
                return data.get("approved", False)
            # 其他消息忽略，继续等待确认
    except (asyncio.TimeoutError, WebSocketDisconnect, json.JSONDecodeError):
        return False


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
