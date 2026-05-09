"""
bmc_agent.config — 配置加载与 LLM 工厂

本模块负责:
    1. 从 config.yaml 加载配置（BMC 连接参数、LLM 配置、服务端口、记忆设置）
    2. 当配置文件不存在或字段缺失时，使用内置默认值填充
    3. 提供 create_llm() 工厂方法，创建适配硅基流动 API 的 ChatOpenAI 实例

配置加载策略:
    - config.yaml 中的值覆盖 DEFAULTS 中的默认值
    - CLI 命令行参数（在 bmc_tool.py / web/app.py 中处理）覆盖 config.yaml
    - 优先级: 命令行 > 配置文件 > 默认值

硅基流动 API 兼容性:
    硅基流动 (SiliconFlow) 提供与 OpenAI 兼容的 API 接口，
    因此使用 langchain_openai.ChatOpenAI，只需修改 base_url 和 api_key。
    支持的模型包括 Qwen、DeepSeek、GLM 等主流开源模型。
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
        "base_url": "https://api.siliconflow.cn/v1",  # 硅基流动 API 地址
        "model": "Qwen/Qwen2.5-7B-Instruct",           # 默认模型
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
    },
    "memory": {
        "backend": "sqlite",      # 记忆后端: sqlite（持久化）或 memory（内存）
        "db_path": "bmc_memory.db",
    },
}


def load_config(path=None) -> dict:
    """加载配置文件，缺失字段用默认值填充。

    Args:
        path: 配置文件路径。None 则使用默认路径（项目根目录/config.yaml）

    Returns:
        dict: 合并后的完整配置字典，结构:
            {
                "bmc": {host, username, password, interface, port, timeout, retries},
                "llm": {api_key, base_url, model},
                "server": {host, port},
                "memory": {backend, db_path},
            }

    加载逻辑:
        1. 以 DEFAULTS 的深拷贝作为基础
        2. 如果配置文件存在，读取 YAML 并逐 section 覆盖默认值
        3. 缺失的配置文件不会报错，仅使用默认值
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(path)

    # 以默认值为基础
    config = {}
    for section, defaults in DEFAULTS.items():
        config[section] = dict(defaults)

    # 配置文件覆盖默认值
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
    完全兼容 OpenAI API 协议格式，支持 Function Calling（工具调用）。

    Args:
        cfg: 完整配置字典，需包含 cfg["llm"] 子配置

    Returns:
        ChatOpenAI: 配置好的 LLM 实例，可直接调用 .bind_tools() 或 .invoke()

    示例:
        llm = create_llm(config)
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
        response = llm_with_tools.invoke([HumanMessage(content="你好")])
    """
    llm_cfg = cfg["llm"]
    return ChatOpenAI(
        api_key=llm_cfg["api_key"],         # 硅基流动 API Key
        base_url=llm_cfg["base_url"],       # 硅基流动 API 地址
        model=llm_cfg["model"],             # 模型名称（如 Pro/zai-org/GLM-5.1）
    )
