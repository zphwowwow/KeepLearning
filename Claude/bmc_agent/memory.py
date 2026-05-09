"""
bmc_agent.memory — 对话记忆持久化

本模块提供 LangGraph 的检查点（Checkpoint）存储器，用于实现对话记忆功能。

记忆机制:
    LangGraph 的 checkpointer 在图的每一步执行后自动保存状态快照，
    包括消息历史、工具调用及结果。当使用相同的 thread_id 再次调用时，
    Agent 可以恢复之前的上下文，实现多轮连续对话。

支持的存储后端:
    - SQLite (推荐): 持久化到磁盘，服务重启后记忆不丢失
    - Memory: 仅存在内存中，服务重启后丢失（适合调试）

使用方式:
    SQLite:
        with get_sqlite_checkpointer("bmc_memory.db") as checkpointer:
            checkpointer.setup()  # 初始化数据库表
            graph = build_graph(llm, checkpointer=checkpointer)
            # ... 使用 graph ...

    Memory:
        checkpointer = get_memory_saver()
        graph = build_graph(llm, checkpointer=checkpointer)

技术细节:
    SqliteSaver.from_conn_string() 返回一个上下文管理器（context manager），
    这是因为 SQLite 连接需要在使用完毕后正确关闭。在 Web 应用中，
    通过 FastAPI 的 lifespan 机制管理其生命周期。
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def get_memory_saver():
    """创建内存型检查点存储器。

    特点:
        - 零配置，开箱即用
        - 服务重启后记忆丢失
        - 适合开发调试

    Returns:
        MemorySaver: 内存存储器实例
    """
    return MemorySaver()


def get_sqlite_checkpointer(db_path: str = "bmc_memory.db"):
    """创建 SQLite 持久化检查点存储器。

    特点:
        - 数据持久化到磁盘，服务重启后记忆保留
        - 支持并发读取
        - 适合生产环境

    使用方式（上下文管理器）:
        with get_sqlite_checkpointer("bmc.db") as checkpointer:
            checkpointer.setup()   # 首次使用需初始化表结构
            graph = build_graph(llm, checkpointer=checkpointer)

    Args:
        db_path: SQLite 数据库文件路径。
                 特殊值 ":memory:" 表示使用内存数据库（不持久化）

    Returns:
        上下文管理器，yield SqliteSaver 实例
    """
    return SqliteSaver.from_conn_string(db_path)
