"""
bmc_agent.graph — LangGraph 状态图（Agent 核心逻辑）

本模块是整个 BMC 管理 Agent 的核心，使用 LangGraph 框架构建了一个
"ReAct (Reasoning + Acting)" 模式的状态图，实现了 LLM 自主选择工具、
执行工具、解读结果、继续推理的循环。

状态图结构:

    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   START ──→ agent ──→ 有 tool_calls? ──→ tools   │
    │               ↑           │                      │
    │               │           ↓                      │
    │               │          END                     │
    │               │                                  │
    │               └───── results ←───────────────────┘
    │                                                  │
    └──────────────────────────────────────────────────┘

执行流程:
    1. 用户消息进入 agent 节点
    2. LLM 根据用户意图 + 工具描述，决定是直接回复还是调用工具
    3. 如果调用工具 → 进入 tools 节点执行 → 结果返回 agent → LLM 再次推理
    4. 如果不需要工具 → LLM 直接回复 → 到达 END

这种设计支持多轮工具调用，例如:
    用户: "服务器关了，帮我打开"
    → LLM 调用 power_status() → 发现确实关机
    → LLM 调用 power_on() → 开机成功
    → LLM 回复: "已经帮你开机了"

checkpointer (记忆):
    当传入 checkpointer 时，LangGraph 会自动保存每轮对话的状态，
    包括消息历史和工具调用结果。下次使用相同 thread_id 调用时，
    Agent 可以"记住"之前的对话上下文。

关键技术:
    - MessagesState: LangGraph 内置的消息状态类型，自动管理消息列表
    - ToolNode: LangGraph 预构建的工具执行节点，自动解析 AIMessage.tool_calls
      并调用对应的 @tool 函数
    - conditional_edges: 条件边，根据 LLM 输出决定下一步走向
"""

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from .tools import ALL_TOOLS

SYSTEM_PROMPT = """\
你是一个 BMC 远程管理助手，通过调用 ipmitool 工具来帮用户管理服务器。
用户会以自然语言描述操作意图，你需要选择合适的工具执行，并用中文解释结果。

注意事项：
- 危险操作（关机、重启、清日志等）执行前必须先向用户确认
- 需要参数但用户未提供的，请向用户询问
- 操作结果用简洁中文回复
- 如果工具返回错误信息，向用户解释错误原因并建议排查方向
"""


def build_graph(llm, checkpointer=None):
    """构建 LangGraph 状态图。

    Args:
        llm: LangChain ChatModel 实例（如 ChatOpenAI），需支持 tool calling
        checkpointer: 可选的状态检查点存储器，传入后启用对话记忆功能。
                      支持 SqliteSaver（持久化到磁盘）或 MemorySaver（内存）。

    Returns:
        CompiledGraph: 编译后的 LangGraph 图，可调用 .invoke() 或 .stream()

    使用示例:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", api_key="...", base_url="...")
        graph = build_graph(llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content="查看电源状态")]},
            config={"configurable": {"thread_id": "user-123"}},
        )
    """
    # 将所有工具绑定到 LLM，使 LLM 能在回复中生成 tool_calls
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: MessagesState):
        """Agent 节点: 调用 LLM 进行推理。

        每次进入此节点时:
            1. 将 SYSTEM_PROMPT + 历史消息传给 LLM
            2. LLM 返回 AIMessage，可能包含 tool_calls
            3. 返回新消息追加到状态中

        Args:
            state: 当前消息状态，包含所有历史消息

        Returns:
            dict: {"messages": [AIMessage]} 追加到消息列表
        """
        # 每次调用都注入 system prompt，确保 LLM 角色一致
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        messages = [system_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ToolNode 是 LangGraph 预构建的工具执行节点
    # 它会自动解析 AIMessage.tool_calls，调用对应的 @tool 函数，
    # 并将结果封装为 ToolMessage 返回
    tool_node = ToolNode(ALL_TOOLS)

    def should_continue(state: MessagesState):
        """条件边: 判断 LLM 是否需要继续调用工具。

        检查最后一条消息（AIMessage）是否包含 tool_calls:
            - 有 tool_calls → 路由到 "tools" 节点执行工具
            - 无 tool_calls → 路由到 END，返回最终回复

        Args:
            state: 当前消息状态

        Returns:
            str: "tools" 或 END
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # ── 构建状态图 ──────────────────────────────────────────────────
    graph = StateGraph(MessagesState)

    # 添加节点
    graph.add_node("agent", agent_node)    # LLM 推理节点
    graph.add_node("tools", tool_node)      # 工具执行节点

    # 定义边
    graph.add_edge(START, "agent")          # 入口 → agent
    graph.add_conditional_edges(            # agent → 条件判断 → tools 或 END
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")        # tools → agent（工具结果回到 LLM）

    # 编译图，可选传入 checkpointer 启用记忆功能
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return graph.compile(**compile_kwargs)
