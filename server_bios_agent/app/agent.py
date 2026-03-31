import json
import requests
from typing import List, Dict, Any
from app.config import Config
from app.tools import TOOLS, get_tool_by_name

# ========== 公共函数 ==========
def build_system_prompt_react() -> str:
    """手动 ReAct 模式的 system prompt"""
    tools_desc = "\n".join([
        f"- {t['name']}: {t['description']}. Parameters: {json.dumps(t['parameters'])}"
        for t in TOOLS
    ])
    return f"""You are an AI agent that controls servers via OpenBMC. You have access to the following tools:
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

def build_system_prompt_native() -> str:
    """原生 Tool Call 模式的 system prompt（简单引导）"""
    return "You are an AI agent that controls servers via OpenBMC. Use the provided tools when appropriate."

def execute_tool(tool_name: str, params: dict) -> dict:
    tool = get_tool_by_name(tool_name)
    if not tool:
        return {"error": f"Tool {tool_name} not found"}
    try:
        return tool["func"](params)
    except Exception as e:
        return {"error": str(e)}

# ========== ReAct 模式 Agent ==========
class ReActAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = [{"role": "system", "content": build_system_prompt_react()}]

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_name: str, result: dict):
        self.messages.append({
            "role": "assistant",
            "content": f"[Tool {tool_name} result]: {json.dumps(result)}"
        })

    def call_ollama(self, messages: List[Dict]) -> str:
        url = f"{Config.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": Config.MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as e:
            return f"Error calling Ollama: {e}"

    def run(self, user_input: str, max_loops: int = Config.MAX_AGENT_LOOPS) -> str:
        self.add_user_message(user_input)
        loop = 0
        while loop < max_loops:
            loop += 1
            content = self.call_ollama(self.messages)
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

                result = execute_tool(action, params)
                self.add_tool_result(action, result)
            else:
                self.add_assistant_message("I need to output either a Final Answer or an Action. Please respond with the correct format.")
        return "Max loops reached without final answer."

# ========== 原生 Tool Call 模式 Agent ==========
class NativeToolAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = [{"role": "system", "content": build_system_prompt_native()}]

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def call_ollama_with_tools(self, messages) -> Dict[str, Any]:
        """调用 Ollama 并使用原生 tools 参数"""
        # 构建工具列表（OpenAPI 格式）
        tools = []
        for t in TOOLS:
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
            })
        
        url = f"{Config.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": Config.MODEL_NAME,
            "messages": messages,
            "stream": False,
            "tools": tools,
            "options": {"temperature": 0.2}
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def run(self, user_input: str, max_loops: int = Config.MAX_AGENT_LOOPS) -> str:
        self.add_user_message(user_input)
        
        for _ in range(max_loops):
            resp = self.call_ollama_with_tools(self.messages)
            
            if "error" in resp:
                return f"Error: {resp['error']}"
            
            message = resp.get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            # 如果有工具调用
            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]
                    
                    # 执行工具
                    result = execute_tool(tool_name, tool_args)
                    
                    # 添加工具结果消息（Ollama 要求 role='tool' 且包含 tool_call_id）
                    # 注意：Ollama 的 tool 消息格式需要包含 tool_call_id
                    self.messages.append({
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": tc.get("id", "")  # 某些版本可能无 id
                    })
                
                # 继续循环，让模型基于工具结果生成最终答案
                continue
            
            # 没有工具调用，返回内容
            if message.get("content"):
                return message["content"]
        
        return "Max loops reached without final answer."

# ========== 统一入口 ==========
def run_agent(session_id: str, user_input: str) -> str:
    if Config.USE_NATIVE_TOOL_CALL:
        agent = NativeToolAgent(session_id)
    else:
        agent = ReActAgent(session_id)
    return agent.run(user_input)