import json

# 工具1：获取BIOS版本
def tool_get_bios_version(params: dict) -> dict:
    print(f"[TOOL] get_bios_version called with {params}")
    return {"status": "success", "version": "BIOS v2.1.0 (2025-01-15)"}

# 工具2：修改启动顺序
def tool_set_boot_order(params: dict) -> dict:
    print(f"[TOOL] set_boot_order called with {params}")
    order = params.get("order", [])
    return {"status": "success", "message": f"Boot order set to {order}"}

# 工具3：运行压力测试
def tool_run_stress_test(params: dict) -> dict:
    print(f"[TOOL] run_stress_test called with {params}")
    duration = params.get("duration", 60)
    return {"status": "success", "output": f"Stress test ran for {duration}s, no errors."}

# 工具4：抓取系统日志
def tool_capture_logs(params: dict) -> dict:
    print(f"[TOOL] capture_logs called with {params}")
    lines = params.get("lines", 100)
    return {"status": "success", "logs": f"Last {lines} lines of system log:\n... (simulated) ..."}

# 工具5：生成诊断报告
def tool_diagnostic_report(params: dict) -> dict:
    print(f"[TOOL] diagnostic_report called with {params}")
    return {
        "status": "success",
        "report": "## Diagnostic Report\n- BIOS Version: 2.1.0\n- Temperature: 45°C\n- No critical errors found."
    }

# 工具列表（符合MCP规范）
TOOLS = [
    {
        "name": "get_bios_version",
        "description": "Retrieve current BIOS version information.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "func": tool_get_bios_version
    },
    {
        "name": "set_boot_order",
        "description": "Set the boot device order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of boot devices in priority order (e.g., ['hd', 'pxe', 'cdrom'])"
                }
            },
            "required": ["order"]
        },
        "func": tool_set_boot_order
    },
    {
        "name": "run_stress_test",
        "description": "Run a stress test on the server (CPU/memory/IO).",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "integer",
                    "description": "Test duration in seconds",
                    "default": 60
                }
            }
        },
        "func": tool_run_stress_test
    },
    {
        "name": "capture_logs",
        "description": "Capture system or BIOS logs.",
        "parameters": {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to capture",
                    "default": 100
                }
            }
        },
        "func": tool_capture_logs
    },
    {
        "name": "diagnostic_report",
        "description": "Generate a comprehensive diagnostic report.",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "func": tool_diagnostic_report
    }
]

def get_tool_by_name(name: str):
    for t in TOOLS:
        if t["name"] == name:
            return t
    return None

def execute_tool(tool_name: str, params: dict) -> dict:
    tool = get_tool_by_name(tool_name)
    if not tool:
        return {"error": f"Tool {tool_name} not found"}
    try:
        return tool["func"](params)
    except Exception as e:
        return {"error": str(e)}