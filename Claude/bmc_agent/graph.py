"""
bmc_agent.graph — LangGraph 状态图（Agent 核心逻辑）

本模块使用 LangGraph 框架构建 "ReAct (Reasoning + Acting)" 模式的状态图，
实现 LLM 自主选择工具、执行工具、解读结果、继续推理的循环。

增强功能:
    1. 上下文管理 (trim_messages): 滑动窗口 + 摘要压缩，防止 token 超限
    2. 错误重试与降级: API 限流指数退避、网络断开优雅提示
    3. 危险操作确认: SYSTEM_PROMPT 要求 LLM 对危险操作先确认再执行

状态图结构:
    START → agent ──→ 有 tool_calls? ──→ tools ──→ agent
                 │                              ↑
                 └──→ 无 tool_calls ──→ END      │
                     ↑__________________________┘
"""

import time

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from .tools import ALL_TOOLS

SYSTEM_PROMPT = """\
你是一个 BMC 远程管理助手，通过调用 ipmitool 工具来帮用户管理服务器。
用户会以自然语言描述操作意图，你需要选择合适的工具执行，并用中文解释结果。

注意事项：
- 需要参数但用户未提供的，请向用户询问
- 操作结果用简洁中文回复
- 如果工具返回错误信息，向用户解释错误原因并建议排查方向

危险操作确认规则：
- 以下操作属于危险操作，可能导致服务中断或数据丢失：
  power_off（硬关机）、power_reset（硬重置）、power_cycle（电源循环）、sel_clear（清日志）
- 当用户要求执行危险操作时，你不得直接调用工具，必须先回复确认请求
- 确认请求格式："⚠ 即将执行【操作名】，此操作不可撤回，确认执行吗？"
- 只有用户明确回复"确认"/"是"/"y"/"执行"后，才可调用对应工具
- 用户拒绝或表示犹豫时，取消操作并告知已取消
"""

# 摘要压缩用的 system prompt（单次调用，压缩旧对话为摘要）
SUMMARY_PROMPT = """\
请将以下对话历史压缩为一段简洁的摘要，保留关键信息：
- 用户做了什么操作
- 工具返回了什么重要结果
- 用户当前关注的上下文

摘要用中文，控制在 200 字以内，不要包含系统指令内容。"""


def _summarize_messages(llm, messages: list) -> str:
    """使用 LLM 将旧消息压缩为摘要。

    只在消息数量超过阈值时调用，代价是一次额外 LLM 请求。
    压缩后旧消息被替换为一条摘要 SystemMessage，大幅减少 token 消耗。

    Args:
        llm: LangChain ChatModel 实例
        messages: 需要压缩的旧消息列表

    Returns:
        str: 压缩后的摘要文本
    """
    # 构建摘要请求消息
    content_parts = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        text = getattr(msg, "content", "")
        if text:
            content_parts.append(f"[{role}] {text[:500]}")  # 每条截断 500 字符

    combined = "\n".join(content_parts)
    if not combined.strip():
        return ""

    try:
        response = llm.invoke([
            SystemMessage(content=SUMMARY_PROMPT),
            {"role": "user", "content": combined},
        ])
        return response.content
    except Exception:
        # 摘要失败不应阻塞主流程，返回简单截断
        return "（早期对话因上下文管理已被截断）"


def trim_messages(messages: list, llm=None, max_recent: int = 20,
                  enable_summary: bool = True) -> list:
    """上下文管理：滑动窗口 + 摘要压缩。

    策略:
        1. 消息数 <= max_recent → 直接返回，零开销
        2. 消息数 > max_recent → 保留最近 N 条原文，旧消息压缩为摘要

    为什么不直接截断旧消息:
        直接截断会丢失上下文（如"帮我开机"后 LLM 不知道已执行过 power_status），
        摘要压缩保留了关键语义，让 LLM 能理解之前的交互历史。

    Args:
        messages: 完整消息列表
        llm: LangChain ChatModel（摘要压缩用，None 则只做滑动窗口）
        max_recent: 保留最近 N 条消息原文
        enable_summary: 是否启用 LLM 摘要压缩（False 则仅滑动窗口）

    Returns:
        list: 处理后的消息列表
    """
    if len(messages) <= max_recent:
        return messages

    recent = messages[-max_recent:]
    older = messages[:-max_recent]

    if not older:
        return messages

    # 生成摘要
    summary_text = ""
    if enable_summary and llm is not None:
        summary_text = _summarize_messages(llm, older)

    # 组合: [摘要] + [最近 N 条原文]
    result = []
    if summary_text:
        result.append(SystemMessage(content=f"历史对话摘要:\n{summary_text}"))
    result.extend(recent)

    return result


def build_graph(llm, checkpointer=None, context_cfg: dict = None):
    """构建 LangGraph 状态图。

    Args:
        llm: LangChain ChatModel 实例，需支持 tool calling
        checkpointer: 可选的状态检查点存储器
        context_cfg: 上下文管理配置，如 {"max_recent_messages": 20, "enable_summary": True}

    Returns:
        CompiledGraph: 编译后的 LangGraph 图
    """
    # 上下文管理配置
    max_recent = 20
    enable_summary = True
    if context_cfg:
        max_recent = context_cfg.get("max_recent_messages", 20)
        enable_summary = context_cfg.get("enable_summary", True)

    # 将所有工具绑定到 LLM
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # LLM 调用最大重试次数
    MAX_RETRIES = 3

    def agent_node(state: MessagesState):
        """Agent 节点: 调用 LLM 进行推理，包含上下文管理和错误重试。

        处理流程:
            1. trim_messages() 管理上下文长度
            2. 注入 SYSTEM_PROMPT
            3. 调用 LLM，带指数退避重试
            4. 网络/API 错误降级为友好提示消息
        """
        # 上下文管理: 裁剪消息列表，防止 token 超限
        messages = trim_messages(
            state["messages"],
            llm=llm if enable_summary else None,
            max_recent=max_recent,
            enable_summary=enable_summary,
        )

        # 每次调用注入 system prompt
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        full_messages = [system_msg] + messages

        # 带重试的 LLM 调用
        for attempt in range(MAX_RETRIES):
            try:
                response = llm_with_tools.invoke(full_messages)
                return {"messages": [response]}

            except Exception as e:
                error_name = type(e).__name__
                error_msg = str(e)

                # 限流错误 (429): 指数退避等待后重试
                if "429" in error_msg or "rate" in error_name.lower() or "rate_limit" in error_name.lower():
                    wait = min(2 ** attempt, 16)  # 1s, 2s, 4s, 最大 16s
                    time.sleep(wait)
                    continue

                # 认证错误 (401/403): 不重试，直接返回
                if "401" in error_msg or "403" in error_msg or "auth" in error_name.lower():
                    return {"messages": [AIMessage(
                        content="⚠ API 认证失败，请检查 config.yaml 中的 api_key 是否正确"
                    )]}

                # 网络错误: 不重试（可能持续断网）
                if "connection" in error_name.lower() or "connect" in error_msg.lower():
                    return {"messages": [AIMessage(
                        content="⚠ API 连接失败，请检查网络连接和 base_url 配置"
                    )]}

                # 上下文超长 (400/token limit): 尝试更激进的裁剪
                if "token" in error_msg.lower() or "context" in error_msg.lower() or "400" in error_msg:
                    # 减半保留量后重试一次
                    if attempt == 0:
                        short_messages = trim_messages(
                            state["messages"],
                            llm=None,
                            max_recent=max_recent // 2,
                            enable_summary=False,
                        )
                        full_messages = [system_msg] + short_messages
                        continue
                    return {"messages": [AIMessage(
                        content="⚠ 对话过长超出模型限制，请开启新会话继续"
                    )]}

                # 其他错误: 重试
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue

                # 重试耗尽
                return {"messages": [AIMessage(
                    content=f"⚠ AI 服务异常 ({error_name}): {error_msg[:200]}"
                )]}

    # ToolNode: 自动解析 tool_calls 并执行
    tool_node = ToolNode(ALL_TOOLS)

    def should_continue(state: MessagesState):
        """条件边: 判断 LLM 是否需要继续调用工具。"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # ── 构建状态图 ──────────────────────────────────────────────────
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return graph.compile(**compile_kwargs)
