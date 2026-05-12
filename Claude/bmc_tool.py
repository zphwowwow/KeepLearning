#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bmc_tool — BMC 远程管理助手 CLI 入口

本脚本是系统的命令行入口，提供了与 Web 界面功能相同的对话式 BMC 管理能力，
但通过终端交互而非浏览器。适合在 SSH 远程会话、自动化脚本等场景下使用。

使用方式:
    # 使用 config.yaml 中的默认配置
    python bmc_tool.py

    # 通过命令行覆盖配置
    python bmc_tool.py --host 10.0.0.1 --username admin --model Qwen/Qwen2.5-72B-Instruct

    # 禁用记忆（不保存对话历史）
    python bmc_tool.py --no-memory

    # 指定配置文件
    python bmc_tool.py -c /path/to/my_config.yaml

    # 启用 ipmitool 详细输出（调试网络问题）
    python bmc_tool.py -v

优先级: 命令行参数 > config.yaml > 内置默认值
"""

import argparse
import getpass
import sys

from bmc_agent.config import load_config, create_llm
from bmc_agent.graph import build_graph
from bmc_agent.manager import BMCManager
from bmc_agent.memory import get_checkpointer, get_sqlite_checkpointer
from bmc_agent.runner import IPMIToolRunner
from bmc_agent.tools import set_manager


def main():
    """CLI 主入口: 解析参数 → 初始化组件 → 进入对话循环。"""
    parser = argparse.ArgumentParser(
        prog="bmc_tool",
        description="BMC 远程管理助手 (对话式 AI Agent)",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径")
    parser.add_argument("--host", "-H", default=None, help="BMC 地址")
    parser.add_argument("--username", "-U", default=None, help="BMC 用户名")
    parser.add_argument("--password", "-P", default=None, help="BMC 密码")
    parser.add_argument("--model", "-m", default=None, help="LLM 模型名称")
    parser.add_argument("--no-memory", action="store_true", help="禁用对话记忆功能")
    parser.add_argument("--verbose", "-v", action="store_true", help="ipmitool 详细输出")
    args = parser.parse_args()

    # ── 配置加载与合并 ──────────────────────────────────────────────
    config = load_config(args.config)

    # 命令行参数覆盖配置文件值
    if args.host:
        config["bmc"]["host"] = args.host
    if args.username:
        config["bmc"]["username"] = args.username
    if args.password:
        config["bmc"]["password"] = args.password
    if args.model:
        config["llm"]["model"] = args.model

    # 必要参数校验与交互式输入
    if not config["bmc"]["host"]:
        print("错误: 未指定 BMC 地址，请在 config.yaml 或 --host 中设置", file=sys.stderr)
        sys.exit(1)
    if not config["bmc"]["password"]:
        config["bmc"]["password"] = getpass.getpass("BMC 密码: ")
    if not config["llm"]["api_key"]:
        config["llm"]["api_key"] = getpass.getpass("API Key: ")

    # ── 组件初始化 ──────────────────────────────────────────────────
    llm = create_llm(config)

    runner = IPMIToolRunner(
        host=config["bmc"]["host"],
        username=config["bmc"]["username"],
        password=config["bmc"]["password"],
        interface=config["bmc"]["interface"],
        port=config["bmc"]["port"],
        timeout=config["bmc"]["timeout"],
        retries=config["bmc"]["retries"],
        verbose=args.verbose,
    )
    manager = BMCManager(runner)
    set_manager(manager)

    # ── 构建图 & 对话循环 ───────────────────────────────────────────
    # 根据是否启用记忆，决定是否使用 SQLite checkpointer
    if args.no_memory:
        graph = build_graph(llm, context_cfg=config.get("context", {}))
        _chat_loop(graph)
    else:
        checkpointer_result = get_checkpointer(config)
        # SQLite 返回上下文管理器，Redis 返回直接实例
        if hasattr(checkpointer_result, "__enter__"):
            with checkpointer_result as checkpointer:
                checkpointer.setup()
                graph = build_graph(llm, checkpointer=checkpointer,
                                    context_cfg=config.get("context", {}))
                _chat_loop(graph)
        else:
            graph = build_graph(llm, checkpointer=checkpointer_result,
                                context_cfg=config.get("context", {}))
            _chat_loop(graph)


def _chat_loop(graph):
    """CLI 对话循环。

    不断读取用户输入，调用 LangGraph 执行 Agent 推理，
    打印工具调用过程和 AI 回复。

    Args:
        graph: 编译后的 LangGraph 状态图
    """
    print(f"\n=== BMC 远程管理助手 ===")
    print("输入自然语言指令，输入 quit 退出\n")

    thread_id = "cli"  # 固定 thread_id，CLI 模式下所有对话共享历史
    from langchain_core.messages import HumanMessage

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        # 调用 LangGraph 执行 Agent
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as e:
            print(f"错误: {e}")
            continue

        # 遍历结果消息，打印工具调用和 AI 回复
        for msg in result["messages"]:
            # 打印工具调用信息
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  🔧 调用: {tc['name']}({tc['args']})")
            # 打印 AI 最终回复（跳过带 tool_calls 的中间回复）
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                if not msg.tool_calls:
                    print(f"助手: {msg.content}\n")


if __name__ == "__main__":
    main()
