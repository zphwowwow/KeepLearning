import asyncio
import json
import requests
from fastmcp import Client

# ================== 配置区域 ==================
MCP_SERVER_URL = "http://localhost:8080/sse"   # MCP 服务器 SSE 端点
OLLAMA_MODEL = "qwen2.5:7b"                    # 支持 tool calling 的模型
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MAX_TOOL_ITERATIONS = 5                        # 最大工具调用轮数，防止无限循环
# =============================================

async def call_mcp_tool(client, tool_name: str, arguments: dict) -> str:
    """调用 MCP 服务器上的工具"""
    result = await client.call_tool(tool_name, arguments)
    return result.content[0].text

async def main():
    print(f"正在连接 MCP 服务器: {MCP_SERVER_URL}")
    async with Client(MCP_SERVER_URL) as mcp_client:
        # 获取工具列表
        tools = await mcp_client.list_tools()
        print(f"发现 {len(tools)} 个工具:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        # 转换为 Ollama 的 tools 格式
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        # 对话历史
        messages = [
            {"role": "system", "content": "你是一个智能助手，可以调用工具来帮助用户。当工具调用失败时，你可以根据错误信息重试或调整参数。"}
        ]

        print("\n" + "="*50)
        print("对话已启动，输入 'exit' 退出")
        print("="*50 + "\n")

        while True:
            user_input = input("\n👤 用户: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            messages.append({"role": "user", "content": user_input})

            # 多轮工具调用循环
            for iteration in range(MAX_TOOL_ITERATIONS):
                # 请求 Ollama（始终携带 tools）
                payload = {
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "tools": ollama_tools,
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
                try:
                    response = requests.post(OLLAMA_API_URL, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    print(result)
                except Exception as e:
                    print(f"❌ 请求 Ollama 失败: {e}")
                    break

                assistant_msg = result["message"]
                messages.append(assistant_msg)

                # 检查是否有工具调用
                if "tool_calls" in assistant_msg and assistant_msg["tool_calls"]:
                    print(f"\n🔧 第 {iteration+1} 轮工具调用:")
                    for tc in assistant_msg["tool_calls"]:
                        tool_name = tc["function"]["name"]
                        args_raw = tc["function"]["arguments"]
                        if isinstance(args_raw, str):
                            arguments = json.loads(args_raw)
                        else:
                            arguments = args_raw
                        print(f"  调用: {tool_name}({arguments})")

                        try:
                            tool_result = await call_mcp_tool(mcp_client, tool_name, arguments)
                            print(f"  返回: {tool_result[:100]}...")  # 截断显示
                        except Exception as e:
                            tool_result = f"工具调用失败: {e}"
                            print(f"  ❌ 失败: {e}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_result
                        })
                    # 继续下一轮，让模型处理工具结果
                    continue
                else:
                    # 没有工具调用，直接输出最终回答
                    answer = assistant_msg.get("content", "")
                    print(f"\n🤖 回答: {answer}")
                    break
            else:
                # 达到最大迭代次数仍未结束
                print("\n⚠️ 达到最大工具调用轮数，停止。")
                # 可选：添加一条提示消息到历史，避免下次对话异常
                messages.append({"role": "assistant", "content": "抱歉，工具调用次数过多，请简化请求或稍后再试。"})
                continue

            # 如果循环正常结束（break），messages 已经包含了助手回答，无需额外操作

if __name__ == "__main__":
    asyncio.run(main())