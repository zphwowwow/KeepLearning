import json

def tool_power_control(params: dict) -> dict:
    print(f"[TOOL] power_control called with {params}")
    action = params.get("action")
    return {"status": "success", "message": f"Power {action} command sent."}

def tool_sol_console(params: dict) -> dict:
    print(f"[TOOL] sol_console called with {params}")
    return {"status": "success", "output": "[SOL] BIOS POST complete. OS booting..."}

def tool_bios_config(params: dict) -> dict:
    print(f"[TOOL] bios_config called with {params}")
    setting = params.get("setting")
    value = params.get("value")
    return {"status": "success", "message": f"BIOS setting {setting} set to {value}."}

def tool_sensor_read(params: dict) -> dict:
    print(f"[TOOL] sensor_read called with {params}")
    return {"status": "success", "data": {"CPU_Temp": "45°C", "Fan1": "2200 RPM"}}

def tool_event_log(params: dict) -> dict:
    print(f"[TOOL] event_log called with {params}")
    return {"status": "success", "entries": ["Event 1: System boot", "Event 2: No errors"]}

def tool_firmware_update(params: dict) -> dict:
    print(f"[TOOL] firmware_update called with {params}")
    return {"status": "success", "message": "Firmware updated to version 2.0.0"}

def tool_stress_test(params: dict) -> dict:
    print(f"[TOOL] stress_test called with {params}")
    return {"status": "success", "output": "Stress test passed. No errors."}

def tool_boot_device(params: dict) -> dict:
    print(f"[TOOL] boot_device called with {params}")
    device = params.get("device")
    return {"status": "success", "message": f"Next boot device set to {device}."}

def tool_kvm_video(params: dict) -> dict:
    print(f"[TOOL] kvm_video called with {params}")
    return {"status": "success", "url": "http://bmc-ip/kvm/screenshot.png"}

def tool_diagnostic(params: dict) -> dict:
    print(f"[TOOL] diagnostic called with {params}")
    return {
        "status": "success",
        "health_score": 95,
        "issues": [],
        "suggestions": "System appears healthy."
    }

# 工具列表及元数据
TOOLS = [
    {
        "name": "power_control",
        "description": "Control server power: on, off, reset, status.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["on", "off", "reset", "status"]}
            },
            "required": ["action"]
        },
        "func": tool_power_control
    },
    {
        "name": "sol_console",
        "description": "Start Serial Over LAN console to capture BIOS boot logs.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {"type": "integer", "description": "Seconds to capture"}
            }
        },
        "func": tool_sol_console
    },
    {
        "name": "bios_config",
        "description": "Get or set BIOS configuration.",
        "parameters": {
            "type": "object",
            "properties": {
                "setting": {"type": "string"},
                "value": {"type": "string"}
            },
            "required": ["setting", "value"]
        },
        "func": tool_bios_config
    },
    {
        "name": "sensor_read",
        "description": "Read BMC sensors (temp, fan, voltage).",
        "parameters": {
            "type": "object",
            "properties": {
                "sensor_name": {"type": "string", "description": "Optional sensor name"}
            }
        },
        "func": tool_sensor_read
    },
    {
        "name": "event_log",
        "description": "Retrieve System Event Log (SEL).",
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "Optional filter"}
            }
        },
        "func": tool_event_log
    },
    {
        "name": "firmware_update",
        "description": "Update BMC/BIOS firmware.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["bmc", "bios"]},
                "url": {"type": "string"}
            },
            "required": ["type", "url"]
        },
        "func": tool_firmware_update
    },
    {
        "name": "stress_test",
        "description": "Run stress test on server (requires OS SSH).",
        "parameters": {
            "type": "object",
            "properties": {
                "test_type": {"type": "string", "enum": ["cpu", "memory", "io"]},
                "duration": {"type": "integer"}
            },
            "required": ["test_type", "duration"]
        },
        "func": tool_stress_test
    },
    {
        "name": "boot_device",
        "description": "Set next boot device.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {"type": "string", "enum": ["pxe", "hd", "cdrom"]}
            },
            "required": ["device"]
        },
        "func": tool_boot_device
    },
    {
        "name": "kvm_video",
        "description": "Get KVM screenshot or video stream.",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "func": tool_kvm_video
    },
    {
        "name": "diagnostic",
        "description": "Run comprehensive diagnostics and return health score.",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "func": tool_diagnostic
    }
]

def get_tool_by_name(name: str):
    for t in TOOLS:
        if t["name"] == name:
            return t
    return None