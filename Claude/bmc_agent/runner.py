"""
bmc_agent.runner — IPMI 底层命令执行器

本模块是整个 BMC 管理系统的最底层，负责通过 subprocess 调用 ipmitool.exe
与远程 BMC (Baseboard Management Controller) 进行通信。

核心设计思路:
    1. 使用 -E 标志 + IPMI_PASSWORD 环境变量传递密码，避免密码出现在
       进程命令行中（-P 标志会暴露密码，可通过 ps / 任务管理器看到）
    2. 设置 cwd 为 ipmitool.exe 所在目录，确保 Windows 能找到同目录下
       的 Cygwin 依赖 DLL（cygwin1.dll, cygcrypto-1.0.0.dll 等）
    3. 所有命令以列表形式传递给 subprocess.run()，不使用 shell=True，
       防止命令注入漏洞
    4. IPMI 网络超时（-N）与 subprocess 超时分离，subprocess 超时作为
       安全兜底，避免因网络卡死导致进程永久挂起

依赖关系:
    ipmitool.exe (Cygwin build) + 5 个 Cygwin DLL → 本模块 → manager.py
"""

import os
import subprocess
from pathlib import Path


# ipmitool.exe 的路径：相对于本文件向上两层，再进入 ipmitool 目录
# 项目结构: CursorProject/bmc_agent/runner.py
#           CursorProject/ipmitool/ipmitool.exe
IPMITOOL_EXE = Path(__file__).resolve().parent.parent / "ipmitool" / "ipmitool.exe"


class IPMIToolError(Exception):
    """IPMI 命令执行失败时抛出的自定义异常。

    可能的触发场景:
        - ipmitool.exe 未找到（FileNotFoundError）
        - 命令执行超时（TimeoutExpired）
        - ipmitool 返回非零退出码（连接失败、认证失败、命令不支持等）
    """
    pass


class IPMIToolRunner:
    """IPMI 命令执行器 —— 封装 ipmitool.exe 的 subprocess 调用。

    职责:
        - 构建完整的 ipmitool 命令行参数（连接参数 + 子命令）
        - 执行命令并处理超时、文件缺失、非零退出码等异常
        - 对外提供 run() 和 run_output() 两个核心方法

    典型调用流程:
        runner = IPMIToolRunner(host="10.8.148.125", username="Admin", password="xxx")
        result = runner.run_output("power", "status")   # → "Chassis Power is on"

    命令行等价:
        ipmitool -I lanplus -H 10.8.148.125 -p 623 -U Admin -E -N 10 -R 2 power status

    Attributes:
        host: BMC 的 IP 地址或主机名
        username: BMC 登录用户名
        password: BMC 登录密码（通过环境变量传递，不出现在命令行中）
        interface: IPMI 接口类型，lanplus（IPMI v2.0/RMCP+）或 lan（v1.5）
        port: BMC 服务端口，默认 623（RMCP 标准端口）
        timeout: IPMI 网络操作超时（秒），传递给 ipmitool -N 参数
        retries: IPMI 命令重试次数，传递给 ipmitool -R 参数
        verbose: 是否启用 ipmitool 的 -v 详细输出模式（调试用）
    """

    def __init__(self, host: str, username: str, password: str,
                 interface: str = "lanplus", port: int = 623,
                 timeout: int = 10, retries: int = 2, verbose: bool = False):
        self.host = host
        self.username = username
        self.password = password
        self.interface = interface
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose

    def _build_base_args(self) -> list:
        """构建 ipmitool 的公共连接参数。

        生成的参数列表:
            [ipmitool.exe路径, -I, lanplus, -H, host, -p, port,
             -U, username, -E, -N, timeout, -R, retries, (-v)]

        关键参数说明:
            -E: 从 IPMI_PASSWORD 环境变量读取密码（安全，不在进程列表中暴露）
            -N: 指定 lan/lanplus 协议层超时秒数
            -R: 指定重试次数
            -v: 可选，启用 verbose 模式输出调试信息
        """
        args = [
            str(IPMITOOL_EXE),
            "-I", self.interface,       # 接口类型：lanplus(IPMI v2.0) 或 lan(v1.5)
            "-H", self.host,            # BMC 目标地址
            "-p", str(self.port),       # RMCP 端口
            "-U", self.username,        # 认证用户名
            "-E",                       # 从环境变量读取密码（安全模式）
            "-N", str(self.timeout),    # 网络超时（秒）
            "-R", str(self.retries),    # 重试次数
        ]
        if self.verbose:
            args.append("-v")           # 详细输出模式
        return args

    def run(self, *command_args: str) -> subprocess.CompletedProcess:
        """执行 ipmitool 命令并返回完整的 CompletedProcess 对象。

        执行流程:
            1. 拼接公共连接参数 + 子命令参数
            2. 将密码写入 IPMI_PASSWORD 环境变量（继承当前进程环境）
            3. 设置 cwd 为 ipmitool 所在目录（DLL 依赖解析）
            4. 执行 subprocess.run()，捕获 stdout/stderr
            5. 异常处理：FileNotFoundError / TimeoutExpired / 非零退出码

        Args:
            *command_args: ipmitool 子命令参数，如 ("power", "status")

        Returns:
            subprocess.CompletedProcess: 包含 stdout、stderr、returncode 等

        Raises:
            IPMIToolError: 任何执行失败的情况
        """
        args = self._build_base_args() + list(command_args)

        # 构建子进程环境变量，将密码通过 IPMI_PASSWORD 传递给 ipmitool -E
        env = os.environ.copy()
        env["IPMI_PASSWORD"] = self.password

        # subprocess 超时 = IPMI 自身超时 × (重试次数+1) + 5秒缓冲
        # 这是兜底超时，正常情况下 ipmitool 自己的 -N/-R 参数会先触发
        subprocess_timeout = self.timeout * (self.retries + 1) + 5

        try:
            result = subprocess.run(
                args,
                capture_output=True,            # 捕获 stdout 和 stderr
                text=True,                      # 以文本模式返回（非 bytes）
                encoding="utf-8",               # 指定编码
                errors="replace",               # 无法解码的字符用 � 替换
                timeout=subprocess_timeout,      # 进程级超时
                cwd=str(IPMITOOL_EXE.parent),    # 工作目录设为 ipmitool 所在目录（DLL 解析）
                env=env,                         # 含 IPMI_PASSWORD 的环境变量
            )
        except FileNotFoundError:
            raise IPMIToolError(f"ipmitool.exe not found at {IPMITOOL_EXE}")
        except subprocess.TimeoutExpired:
            raise IPMIToolError(f"Command timed out after {subprocess_timeout}s")

        # ipmitool 返回非零退出码 → 命令执行失败（认证错误、网络不通等）
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise IPMIToolError(
                f"ipmitool exited with code {result.returncode}: {stderr}"
            )
        return result

    def run_output(self, *command_args: str) -> str:
        """执行 ipmitool 命令并返回 stdout（去除首尾空白）。

        这是最常用的方法，BMCManager 中的所有方法都通过它调用。
        仅关心命令输出的场景使用此方法；需要 stderr 或 returncode 时用 run()。

        Args:
            *command_args: ipmitool 子命令参数，如 ("power", "status")

        Returns:
            str: ipmitool 的标准输出（已 trim）

        Raises:
            IPMIToolError: 同 run()
        """
        return self.run(*command_args).stdout.strip()
