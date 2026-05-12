"""
bmc_agent.tools — LangChain Tool 定义层

本模块使用 LangChain 的 @tool 装饰器，将 BMCManager 中的方法注册为
LLM 可调用的工具（Function Calling / Tool Use）。

这是 Agent 架构中的"工具层"，LLM 根据用户意图选择合适的工具，
框架自动解析工具调用的名称和参数，执行后将结果返回给 LLM。

架构位置:
    用户输入 → LLM (选择工具) → 本模块 (执行工具) → BMCManager → IPMIToolRunner → ipmitool.exe
                ↑                                                    |
                └──────────── 工具执行结果 ←─────────────────────────┘

设计要点:
    1. 使用模块级单例 _manager 存储当前 BMCManager 实例，
       通过 set_manager() 在应用启动时注入（依赖注入模式）
    2. _safe() 统一异常处理，将 IPMIToolError 转换为对用户友好的字符串，
       确保 LLM 能理解错误原因并给出建议，而不是直接崩溃
    3. 每个工具的 docstring 既是对 LLM 的说明（决定 LLM 是否选择该工具），
       也是对开发者 的文档
    4. LAN 工具中 channel 参数默认为 1，简化调用，因为绝大多数服务器
       的 BMC 网络都配置在通道 1

工具分类 (24 个):
    - 电源控制 (6): power_status, power_on, power_off, power_reset, power_cycle, power_soft
    - 传感器 (2): sensor_list, sdr_get
    - 系统日志 (4): sel_list, sel_info, sel_clear, sel_get
    - 用户管理 (5): user_list, user_set_name, user_set_password, user_enable, user_disable
    - 网络配置 (4): lan_print, lan_set_ip, lan_set_netmask, lan_set_gateway
    - 机箱状态 (2): chassis_status, chassis_identify
    - FRU 信息 (1): fru_print
"""

from langchain_core.tools import tool

from .manager import BMCManager
from .runner import IPMIToolError


# ── 模块级单例：存储当前 BMCManager 实例 ──────────────────────────────
# 使用全局变量的依赖注入模式：应用启动时调用 set_manager() 注入实例，
# 工具函数内部通过 _mgr() 获取。这样 @tool 函数可以是无状态的纯函数，
# LangGraph 框架能正确序列化和调度它们。
_manager: BMCManager | None = None


def set_manager(mgr: BMCManager):
    """注入 BMCManager 实例。在应用启动时调用一次。

    Args:
        mgr: 已初始化的 BMCManager 实例
    """
    global _manager
    _manager = mgr


def _mgr() -> BMCManager:
    """获取当前 BMCManager 实例。未初始化时抛出 RuntimeError。"""
    if _manager is None:
        raise RuntimeError("BMCManager not initialized, call set_manager() first")
    return _manager


def _safe(func, **kwargs) -> str:
    """统一异常处理的工具执行包装器，含错误分类和友好提示。

    设计意图:
        - 正常执行：返回 ipmitool 的输出
        - 输出为空：返回"操作成功（无输出）"（部分命令如 user_enable 无输出）
        - 执行失败：根据错误类型给出分类提示，帮助 LLM 更好地向用户解释

    这种"异常转字符串"的模式是 Agent 应用中的常见实践：
    LLM 无法处理 Python 异常，但可以解读错误信息文本。
    细致的错误分类让 LLM 能给出更有针对性的排查建议。

    Args:
        func: 要调用的 BMCManager 方法
        **kwargs: 传递给该方法的参数

    Returns:
        str: 命令执行结果或友好错误信息
    """
    try:
        result = func(**kwargs)
        return result if result else "操作成功（无输出）"
    except IPMIToolError as e:
        error_msg = str(e)
        # 超时: BMC 地址不可达或网络不通
        if "timed out" in error_msg or "timeout" in error_msg.lower():
            return (
                "错误: 命令执行超时。可能原因:\n"
                "  1. BMC 地址不可达，请检查 IP 是否正确\n"
                "  2. 网络不通，请确认与 BMC 在同一网段\n"
                "  3. BMC 服务未启动或端口被防火墙拦截"
            )
        # ipmitool.exe 未找到
        if "not found" in error_msg:
            return (
                "错误: ipmitool.exe 未找到。请确认 ipmitool/ 目录下存在 ipmitool.exe 及 Cygwin DLL"
            )
        # 非零退出码: 认证失败、命令不支持等
        if "exited with code" in error_msg:
            # 常见退出码: 1=一般错误, 2=连接错误, 3=认证错误
            if "code 1" in error_msg:
                return (
                    f"错误: BMC 命令执行失败 — {error_msg}\n"
                    "可能原因: 命令不被此 BMC 固件支持，或参数有误"
                )
            if "code 2" in error_msg or "code 3" in error_msg:
                return (
                    f"错误: BMC 连接/认证失败 — {error_msg}\n"
                    "请检查: 用户名密码是否正确、接口类型(lan/lanplus)是否匹配、BMC 是否启用 IPMI"
                )
            return (
                f"错误: BMC 返回错误 — {error_msg}\n"
                "建议: 检查连接参数（用户名、密码、接口类型）是否正确"
            )
        # 其他未分类错误
        return f"错误: {error_msg}"


# ══════════════════════════════════════════════════════════════════════
# 电源控制工具
# ══════════════════════════════════════════════════════════════════════
# 电源操作是最常用的远程管理功能。power_off / power_reset / power_cycle
# 是危险操作，Agent 的 SYSTEM_PROMPT 会要求 LLM 先向用户确认再调用。

@tool
def power_status() -> str:
    """查看服务器电源状态（开机/关机）。

    LLM 调用场景: 用户问"服务器开着吗？""电源状态是什么？"
    返回示例: "Chassis Power is on"
    """
    return _safe(_mgr().power_status)


@tool
def power_on() -> str:
    """远程开机（给服务器上电）。

    LLM 调用场景: 用户说"帮我开机""把服务器打开"
    返回示例: "Chassis Power Control: Up/On"
    """
    return _safe(_mgr().power_on)


@tool
def power_off() -> str:
    """远程关机（直接断电，危险操作，请先确认）。

    LLM 调用场景: 用户说"关机""断电"
    注意: 这是硬关机，不经过操作系统，可能造成数据丢失。
    """
    return _safe(_mgr().power_off)


@tool
def power_reset() -> str:
    """硬重置服务器（危险操作，请先确认）。

    LLM 调用场景: 用户说"重启服务器""硬重启"
    注意: 等效于按下物理复位按钮，不经过 OS 正常关机流程。
    """
    return _safe(_mgr().power_reset)


@tool
def power_cycle() -> str:
    """电源循环——先断电再上电（危险操作，请先确认）。

    LLM 调用场景: 用户说"电源循环""断电重启"
    """
    return _safe(_mgr().power_cycle)


@tool
def power_soft() -> str:
    """软关机（通过 ACPI 正常关机）。

    LLM 调用场景: 用户说"优雅关机""正常关机"
    与 power_off 不同，这会通过 ACPI 信号通知 OS 执行关机流程。
    """
    return _safe(_mgr().power_soft)


# ══════════════════════════════════════════════════════════════════════
# 传感器工具
# ══════════════════════════════════════════════════════════════════════

@tool
def sensor_list() -> str:
    """列出所有传感器及其当前读数。

    LLM 调用场景: 用户说"看看温度""传感器数据""风扇转速"
    返回包含温度、电压、风扇转速等传感器列表。
    """
    return _safe(_mgr().sensor_list)


@tool
def sdr_get(sensor_id: str) -> str:
    """查询指定传感器的详细信息。

    LLM 调用场景: 用户说"CPU温度详情""查看某个传感器"
    Args:
        sensor_id: 传感器 ID 或名称（如 "CPU1 Temp"）
    """
    return _safe(_mgr().sdr_get, sensor_id=sensor_id)


# ══════════════════════════════════════════════════════════════════════
# 系统事件日志 (SEL) 工具
# ══════════════════════════════════════════════════════════════════════

@tool
def sel_list() -> str:
    """查看系统事件日志 (SEL)。

    LLM 调用场景: 用户说"看看日志""有什么告警""事件记录"
    返回 SEL 条目列表，包含时间戳、事件类型、描述。
    """
    return _safe(_mgr().sel_list)


@tool
def sel_info() -> str:
    """查看系统日志信息（条目数、容量等）。

    LLM 调用场景: 用户说"日志有多少条""日志空间够吗"
    """
    return _safe(_mgr().sel_info)


@tool
def sel_clear() -> str:
    """清除系统事件日志（危险操作，不可恢复，请先确认）。

    LLM 调用场景: 用户说"清空日志""清除SEL"
    一旦清除无法恢复，Agent 应先向用户确认。
    """
    return _safe(_mgr().sel_clear)


@tool
def sel_get(record_id: int) -> str:
    """查看指定日志记录详情。

    LLM 调用场景: 用户说"第5条日志的详情""ID为10的记录"
    Args:
        record_id: 日志记录 ID
    """
    return _safe(_mgr().sel_get, record_id=record_id)


# ══════════════════════════════════════════════════════════════════════
# 用户管理工具
# ══════════════════════════════════════════════════════════════════════

@tool
def user_list() -> str:
    """列出 BMC 用户。

    LLM 调用场景: 用户说"有哪些用户""用户列表"
    """
    return _safe(_mgr().user_list)


@tool
def user_set_name(user_id: int, name: str) -> str:
    """设置用户名。

    LLM 调用场景: 用户说"把2号用户改名为monitor"
    Args:
        user_id: 用户 ID（BMC 内部编号）
        name: 新用户名
    """
    return _safe(_mgr().user_set_name, user_id=user_id, name=name)


@tool
def user_set_password(user_id: int, password: str) -> str:
    """设置用户密码。

    LLM 调用场景: 用户说"修改2号用户的密码"
    Args:
        user_id: 用户 ID
        password: 新密码
    """
    return _safe(_mgr().user_set_password, user_id=user_id, password=password)


@tool
def user_enable(user_id: int) -> str:
    """启用指定用户。

    LLM 调用场景: 用户说"启用3号用户"
    Args:
        user_id: 用户 ID
    """
    return _safe(_mgr().user_enable, user_id=user_id)


@tool
def user_disable(user_id: int) -> str:
    """禁用指定用户。

    LLM 调用场景: 用户说"禁用3号用户"
    Args:
        user_id: 用户 ID
    """
    return _safe(_mgr().user_disable, user_id=user_id)


# ══════════════════════════════════════════════════════════════════════
# 网络配置 (LAN) 工具
# ══════════════════════════════════════════════════════════════════════
# LAN 修改操作需谨慎，错误的 IP/网关配置可能导致 BMC 失联。
# channel 参数在此层固定为 1（简化 LLM 调用），绝大多数服务器都使用通道 1。

@tool
def lan_print() -> str:
    """查看 BMC 网络配置（IP、子网掩码、网关等）。

    LLM 调用场景: 用户说"看看BMC的IP""网络配置是什么"
    """
    return _safe(_mgr().lan_print)


@tool
def lan_set_ip(ip_addr: str) -> str:
    """设置 BMC 的 IP 地址。

    LLM 调用场景: 用户说"把BMC的IP改成10.0.0.50"
    Args:
        ip_addr: 新 IP 地址
    """
    return _safe(_mgr().lan_set_ip, channel=1, ip_addr=ip_addr)


@tool
def lan_set_netmask(netmask: str) -> str:
    """设置 BMC 的子网掩码。

    LLM 调用场景: 用户说"设置子网掩码为255.255.255.0"
    Args:
        netmask: 子网掩码
    """
    return _safe(_mgr().lan_set_netmask, channel=1, mask=netmask)


@tool
def lan_set_gateway(gateway: str) -> str:
    """设置 BMC 的默认网关。

    LLM 调用场景: 用户说"设置网关为10.0.0.1"
    Args:
        gateway: 网关 IP 地址
    """
    return _safe(_mgr().lan_set_gateway, channel=1, gw=gateway)


# ══════════════════════════════════════════════════════════════════════
# 机箱状态工具
# ══════════════════════════════════════════════════════════════════════

@tool
def chassis_status() -> str:
    """查看机箱状态（电源、故障指示灯等）。

    LLM 调用场景: 用户说"机箱状态""有没有故障灯亮"
    """
    return _safe(_mgr().chassis_status)


@tool
def chassis_identify(interval: int | None = None) -> str:
    """点亮或闪烁机箱识别指示灯，用于在机房中定位物理服务器。

    LLM 调用场景: 用户说"帮我亮一下指示灯""找到这台机器"
    Args:
        interval: 闪烁间隔秒数，不传则为默认时长
    """
    if interval is not None:
        return _safe(_mgr().chassis_identify, interval=interval)
    return _safe(_mgr().chassis_identify)


# ══════════════════════════════════════════════════════════════════════
# FRU 信息工具
# ══════════════════════════════════════════════════════════════════════

@tool
def fru_print() -> str:
    """查看 FRU (Field Replaceable Unit) 信息，如产品型号、序列号。

    LLM 调用场景: 用户说"看看服务器型号""序列号是什么""硬件信息"
    FRU 信息包含产品制造商、型号、序列号、部件号等，用于资产管理和故障报修。
    """
    return _safe(_mgr().fru_print)


# ══════════════════════════════════════════════════════════════════════
# 工具注册表
# ══════════════════════════════════════════════════════════════════════
# ALL_TOOLS 列表是所有已注册工具的集合，在 graph.py 中被传递给
# LangGraph 的 ToolNode 和 LLM.bind_tools()，使 LLM 能够识别和调用这些工具。

ALL_TOOLS = [
    # 电源控制 (6)
    power_status, power_on, power_off, power_reset, power_cycle, power_soft,
    # 传感器 (2)
    sensor_list, sdr_get,
    # 系统日志 (4)
    sel_list, sel_info, sel_clear, sel_get,
    # 用户管理 (5)
    user_list, user_set_name, user_set_password, user_enable, user_disable,
    # 网络配置 (4)
    lan_print, lan_set_ip, lan_set_netmask, lan_set_gateway,
    # 机箱状态 (2)
    chassis_status, chassis_identify,
    # FRU 信息 (1)
    fru_print,
]
