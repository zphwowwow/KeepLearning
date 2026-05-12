"""
bmc_agent.config — 配置加载与 LLM 工厂

本模块负责:
    1. 从 config.yaml 加载配置（BMC 连接参数、LLM 配置、服务端口、记忆设置等）
    2. 当配置文件不存在或字段缺失时，使用内置默认值填充
    3. 提供 create_llm() 工厂方法，创建适配硅基流动 API 的 ChatOpenAI 实例
    4. 提供 get_checkpointer() 工厂方法，根据配置自动选择记忆后端（SQLite/Redis）

配置加载策略:
    - config.yaml 中的值覆盖 DEFAULTS 中的默认值
    - CLI 命令行参数（在 bmc_tool.py / web/app.py 中处理）覆盖 config.yaml
    - 优先级: 命令行 > 配置文件 > 默认值
"""

from pathlib import Path

import yaml
from langchain_openai import ChatOpenAI

# 默认配置文件路径: 项目根目录下的 config.yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# 内置默认值，确保即使配置文件缺失也能正常运行
DEFAULTS = {
    "bmc": {
        "host": "",
        "username": "admin",
        "password": "",
        "interface": "lanplus",
        "port": 623,
        "timeout": 10,
        "retries": 2,
    },
    "llm": {
        "api_key": "",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
    },
    "memory": {
        "backend": "sqlite",        # sqlite 或 redis
        "db_path": "bmc_memory.db",
        "redis_url": "redis://localhost:6379/0",
    },
    "context": {
        "max_recent_messages": 20,   # 保留最近 N 条消息原文
        "enable_summary": True,      # 是否启用摘要压缩
    },
    "danger": {
        "confirm_required": True,    # 是否启用危险操作人工确认
        "tools": ["power_off", "power_reset", "power_cycle", "sel_clear"],
    },
}


def load_config(path=None) -> dict:
    """加载配置文件，缺失字段用默认值填充。

    Args:
        path: 配置文件路径。None 则使用默认路径

    Returns:
        dict: 合并后的完整配置字典
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(path)

    config = {}
    for section, defaults in DEFAULTS.items():
        config[section] = dict(defaults)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for section in DEFAULTS:
            if section in data:
                config[section].update(data[section])

    return config


def create_llm(cfg: dict) -> ChatOpenAI:
    """根据配置创建 LLM 实例。

    使用 langchain_openai.ChatOpenAI，通过 base_url 指向硅基流动 API，
    完全兼容 OpenAI API 协议格式，支持 Function Calling。

    Args:
        cfg: 完整配置字典，需包含 cfg["llm"] 子配置

    Returns:
        ChatOpenAI: 配置好的 LLM 实例
    """
    llm_cfg = cfg["llm"]
    return ChatOpenAI(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
    )
