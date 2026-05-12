"""
bmc_agent.memory — 对话记忆持久化

本模块提供 LangGraph 的检查点（Checkpoint）存储器，用于实现对话记忆功能。

支持两种存储后端，可在 config.yaml 中通过 memory.backend 切换:
    - SQLite: 持久化到磁盘，服务重启后记忆不丢失，零依赖
    - Redis:  高性能持久化，适合生产环境，需要 Redis 服务

自动降级策略:
    当配置为 Redis 但连接失败时，自动降级到 SQLite，
    确保服务不会因 Redis 不可用而无法启动。

使用方式:
    SQLite (推荐开箱即用):
        with get_sqlite_checkpointer("bmc_memory.db") as cp:
            cp.setup()
            graph = build_graph(llm, checkpointer=cp)

    Redis (需要 Redis 服务):
        cp = get_redis_checkpointer("redis://localhost:6379/0")
        graph = build_graph(llm, checkpointer=cp)

    自动选择 (推荐):
        cp = get_checkpointer(config)  # 根据配置自动选择，Redis 失败自动降级
        graph = build_graph(llm, checkpointer=cp)
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def get_memory_saver():
    """创建内存型检查点存储器（调试用，重启丢失）。"""
    return MemorySaver()


def get_sqlite_checkpointer(db_path: str = "bmc_memory.db"):
    """创建 SQLite 持久化检查点存储器。

    使用上下文管理器模式，确保数据库连接正确释放。

    Args:
        db_path: SQLite 数据库文件路径，":memory:" 为内存模式

    Returns:
        上下文管理器，yield SqliteSaver 实例
    """
    return SqliteSaver.from_conn_string(db_path)


def get_redis_checkpointer(redis_url: str = "redis://localhost:6379/0"):
    """创建 Redis 持久化检查点存储器。

    需要先启动 Redis 服务:
        docker compose up -d redis
        # 或直接: redis-server

    Args:
        redis_url: Redis 连接地址

    Returns:
        RedisSaver 实例

    Raises:
        ImportError: langgraph-checkpoint-redis 未安装
        Exception: Redis 连接失败
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver
    except ImportError:
        raise ImportError(
            "Redis checkpointer 未安装，请运行: pip install langgraph-checkpoint-redis redis"
        )
    return RedisSaver.from_conn_string(redis_url)


def get_checkpointer(cfg: dict):
    """根据配置自动选择记忆后端，Redis 失败时自动降级到 SQLite。

    降级策略:
        1. 配置为 sqlite → 使用 SQLite（上下文管理器模式）
        2. 配置为 redis → 尝试连接 Redis，失败则降级 SQLite
        3. 任何异常 → 降级 SQLite

    Args:
        cfg: 完整配置字典，需包含 cfg["memory"] 子配置

    Returns:
        SqliteSaver 上下文管理器 或 RedisSaver 实例
    """
    backend = cfg["memory"].get("backend", "sqlite")

    if backend == "redis":
        try:
            checkpointer = get_redis_checkpointer(cfg["memory"].get("redis_url", "redis://localhost:6379/0"))
            print(f"[OK] 记忆后端: Redis ({cfg['memory'].get('redis_url', '')})")
            return checkpointer
        except Exception as e:
            print(f"[WARN] Redis 连接失败 ({e})，降级到 SQLite")

    # SQLite 或 Redis 降级
    ctx = get_sqlite_checkpointer(cfg["memory"].get("db_path", "bmc_memory.db"))
    print(f"[OK] 记忆后端: SQLite ({cfg['memory'].get('db_path', 'bmc_memory.db')})")
    return ctx
