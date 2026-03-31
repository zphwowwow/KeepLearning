import requests
from typing import Dict, List, Any

class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def get_tools(self) -> List[Dict]:
        """获取所有可用工具描述"""
        resp = requests.post(f"{self.server_url}/tools/list")
        resp.raise_for_status()
        data = resp.json()
        return data.get("tools", [])

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """调用指定工具"""
        payload = {"name": tool_name, "arguments": arguments}
        resp = requests.post(f"{self.server_url}/tools/call", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result")