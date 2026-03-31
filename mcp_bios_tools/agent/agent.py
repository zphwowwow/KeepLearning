import json
import requests
from typing import Dict, List, Any
from agent.mcp_client import MCPClient

class ReActAgent:
    def __init__(self, mcp_server_url: str, session_id: str = None):
        self.mcp_client = MCPClient(mcp_server_url)
        self.session_id = session_id or "default"
        self.messages = [{"role": "system", "content": self._build_system_prompt()}]
        self.tools = self.mcp_client.get_tools()  # 动态获取工具

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}. Parameters: {json.dumps(t['parameters'])}"
            for t in self.tools
        ])
        return f"""You are an AI agent that controls servers via BIOS tools. You have access to the following tools:
{tools_desc}

You must respond in the following format:
Thought: (your reasoning)
Action: tool_name
Action Input: {{"param": "value"}}

OR if you have the final answer:
Thought: I have enough information.
Final Answer: (your answer)

You can only use tools from the list above. When you get tool results, continue reasoning until you can give a final answer.
"""

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_name: str, result: dict):
        self.messages.append({
            "role": "assistant",
            "content": f"[Tool {tool_name} result]: {json.dumps(result)}"
        })

    def call_llm(self, messages: List[Dict]) -> str:
        """此处应调用本地或远程 LLM，这里用模拟（实际可接入 Ollama 或 OpenAI）"""
        # 简化：直接调用一个本地模型（例如通过 Ollama 的 /api/chat）
        # 此处为示例，实际可替换为对 Ollama 的调用
        # 为了演示，我们返回一个固定的模拟响应（实际项目中应接入真实 LLM）
        # 此处使用 requests 调用本地 Ollama（假设运行在 localhost:11434）
        import requests
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "qwen2.5:7b",
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except Exception as e:
            return f"Error calling LLM: {e}"

    def run(self, user_input: str, max_loops: int = 10) -> str:
        self.add_user_message(user_input)
        loop = 0
        while loop < max_loops:
            loop += 1
            content = self.call_llm(self.messages)
            self.add_assistant_message(content)

            if "Final Answer:" in content:
                final_answer = content.split("Final Answer:", 1)[1].strip()
                return final_answer
            elif "Action:" in content and "Action Input:" in content:
                try:
                    action_line = [l for l in content.split("\n") if l.startswith("Action:")][0]
                    action = action_line.split("Action:", 1)[1].strip()
                    input_line = [l for l in content.split("\n") if l.startswith("Action Input:")][0]
                    input_str = input_line.split("Action Input:", 1)[1].strip()
                    params = json.loads(input_str)
                except Exception as e:
                    self.add_assistant_message(f"Error parsing action: {e}. Please output in correct format.")
                    continue

                # 通过 MCP 调用工具
                result = self.mcp_client.call_tool(action, params)
                self.add_tool_result(action, result)
            else:
                self.add_assistant_message("I need to output either a Final Answer or an Action. Please respond with the correct format.")
        return "Max loops reached without final answer."

def run_agent(mcp_server_url: str, user_input: str) -> str:
    agent = ReActAgent(mcp_server_url)
    return agent.run(user_input)