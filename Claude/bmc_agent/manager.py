"""
bmc_agent.manager — BMC 操作管理器

本模块在 IPMIToolRunner 之上封装了 7 大类共 24 个服务器管理操作，
是 Agent 工具层（tools.py）与底层命令执行器（runner.py）之间的桥梁。

功能模块:
    ┌──────────────┬────────────────────────────────────────────┐
    │ 模块          │ 方法                                       │
    ├──────────────┼────────────────────────────────────────────┤
    │ 电源控制      │ power_status / on / off / reset / cycle / soft │
    │ 传感器读取    │ sensor_list / sdr_list / sdr_get           │
    │ 系统日志 SEL  │ sel_list / info / clear / get              │
    │ 用户管理      │ user_list / summary / set_name / set_password / enable / disable / test │
    │ 网络配置 LAN  │ lan_print / set_ip / set_netmask / set_gateway / set_mac │
    │ 机箱状态      │ chassis_status / chassis_identify          │
    │ FRU 信息      │ fru_print                                 │
    └──────────────┴────────────────────────────────────────────┘

设计原则:
    - 每个方法对应一条 ipmitool 命令，方法命名即命令含义
    - 所有方法返回 str（命令输出）或抛出 IPMIToolError
    - 参数使用 Python 类型（int/str），内部转换为 ipmitool 所需的字符串格式
    - LAN 操作的 channel 参数默认为 1（绝大多数服务器的默认通道）

依赖关系:
    runner.py → 本模块 → tools.py (LangChain @tool)
"""

from typing import Optional

from .runner import IPMIToolRunner


class BMCManager:
    """BMC 操作管理器 —— 将 IPMIToolRunner 的原始命令封装为语义化的 Python 方法。

    每个 public 方法:
        1. 接收 Python 类型的参数（int, str, Optional[int] 等）
        2. 内部将参数转换为 ipmitool 所需的字符串格式
        3. 调用 runner.run_output() 执行命令
        4. 返回命令的 stdout 字符串

    使用示例:
        runner = IPMIToolRunner(host="10.8.148.125", username="Admin", password="xxx")
        mgr = BMCManager(runner)
        mgr.power_status()           # → "Chassis Power is on"
        mgr.lan_set_ip(1, "10.0.0.50")  # → 设置 BMC IP
        mgr.sel_list()               # → 返回 SEL 日志条目

    Attributes:
        _r: IPMIToolRunner 实例，实际执行 ipmitool 命令
    """

    def __init__(self, runner: IPMIToolRunner):
        self._r = runner

    # ── 电源控制 ──────────────────────────────────────────────────────
    # IPMI 电源命令是最常用的远程管理操作，支持开关机、重启、电源循环等。
    # power_off / power_reset / power_cycle 属于高风险操作，在 Agent 层
    # （graph.py 的 SYSTEM_PROMPT）会要求 LLM 先确认再执行。

    def power_status(self) -> str:
        """查看服务器电源状态（开机/关机）。对应: ipmitool power status"""
        return self._r.run_output("power", "status")

    def power_on(self) -> str:
        """远程开机。对应: ipmitool power on"""
        return self._r.run_output("power", "on")

    def power_off(self) -> str:
        """远程硬关机（直接断电，不经过 OS）。对应: ipmitool power off"""
        return self._r.run_output("power", "off")

    def power_reset(self) -> str:
        """硬重置服务器（等效于按下物理复位按钮）。对应: ipmitool power reset"""
        return self._r.run_output("power", "reset")

    def power_cycle(self) -> str:
        """电源循环——先断电再上电。对应: ipmitool power cycle"""
        return self._r.run_output("power", "cycle")

    def power_soft(self) -> str:
        """软关机（通过 ACPI 信号通知 OS 正常关机）。对应: ipmitool power soft"""
        return self._r.run_output("power", "soft")

    # ── 传感器 ────────────────────────────────────────────────────────
    # 传感器数据包括温度、电压、风扇转速等，是服务器健康监控的核心数据源。
    # sensor_list 返回当前值，sdr_get 可获得更详细的传感器元信息。

    def sensor_list(self) -> str:
        """列出所有传感器及其当前读数。对应: ipmitool sensor"""
        return self._r.run_output("sensor")

    def sdr_list(self, sdr_type: str = "all") -> str:
        """列出传感器数据记录(SDR)。
        对应: ipmitool sdr list [type]
        Args:
            sdr_type: SDR 类型过滤，可选 all/full/compact/event 等
        """
        return self._r.run_output("sdr", "list", sdr_type)

    def sdr_get(self, sensor_id: str) -> str:
        """查询指定传感器的详细信息。
        对应: ipmitool sdr get <sensor_id>
        Args:
            sensor_id: 传感器 ID 或名称（如 "CPU1 Temp"）
        """
        return self._r.run_output("sdr", "get", sensor_id)

    # ── 系统事件日志 (SEL) ────────────────────────────────────────────
    # SEL 记录了服务器运行过程中的关键事件（温度告警、电源异常、风扇故障等），
    # 是故障排查的重要数据来源。sel_clear 不可恢复，属于危险操作。

    def sel_list(self) -> str:
        """查看系统事件日志。对应: ipmitool sel list"""
        return self._r.run_output("sel", "list")

    def sel_info(self) -> str:
        """查看日志信息（条目数、空间占用等）。对应: ipmitool sel info"""
        return self._r.run_output("sel", "info")

    def sel_clear(self) -> str:
        """清除系统事件日志（危险操作，不可恢复）。对应: ipmitool sel clear"""
        return self._r.run_output("sel", "clear")

    def sel_get(self, record_id: int) -> str:
        """查看指定日志记录详情。
        对应: ipmitool sel get <id>
        Args:
            record_id: 日志记录 ID
        """
        return self._r.run_output("sel", "get", str(record_id))

    # ── 用户管理 ──────────────────────────────────────────────────────
    # 管理 BMC 上的用户账户，包括查看、创建、启用/禁用、修改密码等。
    # user_id 为 BMC 内部的用户 ID 编号（整数）。

    def user_list(self) -> str:
        """列出 BMC 用户。对应: ipmitool user list"""
        return self._r.run_output("user", "list")

    def user_summary(self) -> str:
        """查看用户概要（最大用户数、已启用数等）。对应: ipmitool user summary"""
        return self._r.run_output("user", "summary")

    def user_set_name(self, user_id: int, name: str) -> str:
        """设置用户名。
        对应: ipmitool user set name <id> <name>
        Args:
            user_id: 用户 ID
            name: 新用户名
        """
        return self._r.run_output("user", "set", "name", str(user_id), name)

    def user_set_password(self, user_id: int, password: str) -> str:
        """设置用户密码。
        对应: ipmitool user set password <id> <password>
        Args:
            user_id: 用户 ID
            password: 新密码
        """
        return self._r.run_output("user", "set", "password", str(user_id), password)

    def user_enable(self, user_id: int) -> str:
        """启用指定用户。对应: ipmitool user enable <id>"""
        return self._r.run_output("user", "enable", str(user_id))

    def user_disable(self, user_id: int) -> str:
        """禁用指定用户。对应: ipmitool user disable <id>"""
        return self._r.run_output("user", "disable", str(user_id))

    def user_test(self, user_id: int, password: str) -> str:
        """测试用户密码是否正确。
        对应: ipmitool user test <id> <password>
        Args:
            user_id: 用户 ID
            password: 待测试密码
        """
        return self._r.run_output("user", "test", str(user_id), password)

    # ── 网络配置 (LAN) ────────────────────────────────────────────────
    # 管理 BMC 的网络配置，包括查看 IP/掩码/网关/MAC，以及修改这些配置。
    # channel 参数对应 IPMI 的逻辑通道号，绝大多数服务器默认使用通道 1。
    # LAN 修改操作需谨慎，错误的 IP 设置可能导致 BMC 失联。

    def lan_print(self, channel: int = 1) -> str:
        """查看 BMC 网络配置（IP、子网掩码、网关、MAC 等）。
        对应: ipmitool lan print <channel>
        Args:
            channel: 通道号，默认 1
        """
        return self._r.run_output("lan", "print", str(channel))

    def lan_set_ip(self, channel: int, ip_addr: str) -> str:
        """设置 BMC 的 IP 地址。
        对应: ipmitool lan set <channel> ipaddr <ip>
        Args:
            channel: 通道号
            ip_addr: 新 IP 地址
        """
        return self._r.run_output("lan", "set", str(channel), "ipaddr", ip_addr)

    def lan_set_netmask(self, channel: int, mask: str) -> str:
        """设置 BMC 的子网掩码。
        对应: ipmitool lan set <channel> netmask <mask>
        Args:
            channel: 通道号
            mask: 子网掩码（如 "255.255.255.0"）
        """
        return self._r.run_output("lan", "set", str(channel), "netmask", mask)

    def lan_set_gateway(self, channel: int, gw: str) -> str:
        """设置 BMC 的默认网关。
        对应: ipmitool lan set <channel> defgw ipaddr <gateway>
        Args:
            channel: 通道号
            gw: 网关 IP 地址
        """
        return self._r.run_output("lan", "set", str(channel), "defgw", "ipaddr", gw)

    def lan_set_mac(self, channel: int, mac: str) -> str:
        """设置 BMC 的 MAC 地址。
        对应: ipmitool lan set <channel> macaddr <mac>
        Args:
            channel: 通道号
            mac: MAC 地址（如 "00:1a:2b:3c:4d:5e"）
        """
        return self._r.run_output("lan", "set", str(channel), "macaddr", mac)

    # ── 机箱状态 ──────────────────────────────────────────────────────
    # chassis_status 返回电源状态、故障指示灯、驱动器等综合信息。
    # chassis_identify 用于远程点亮机箱指示灯，方便在机房中定位物理服务器。

    def chassis_status(self) -> str:
        """查看机箱状态（电源、故障指示灯等）。对应: ipmitool chassis status"""
        return self._r.run_output("chassis", "status")

    def chassis_identify(self, interval: Optional[int] = None) -> str:
        """点亮或闪烁机箱识别指示灯，用于在机房中定位物理服务器。
        对应: ipmitool chassis identify [interval]
        Args:
            interval: 闪烁间隔秒数。None = 使用默认时长，0 = 永久点亮
        """
        if interval is not None:
            return self._r.run_output("chassis", "identify", str(interval))
        return self._r.run_output("chassis", "identify")

    # ── FRU 信息 ──────────────────────────────────────────────────────
    # FRU (Field Replaceable Unit) 信息包含产品型号、序列号、制造商等，
    # 在硬件资产管理、故障报修场景中非常重要。

    def fru_print(self, fru_id: Optional[int] = None) -> str:
        """查看 FRU 信息（产品型号、序列号等）。
        对应: ipmitool fru print [fru_id]
        Args:
            fru_id: FRU ID，不传则显示所有 FRU 设备
        """
        if fru_id is not None:
            return self._r.run_output("fru", "print", str(fru_id))
        return self._r.run_output("fru", "print")
