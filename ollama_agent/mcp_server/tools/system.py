import subprocess
import platform

def restart_computer(confirm: bool = False, delay: int = 0) -> str:
    """
    重启计算机（需要管理员权限）。
    注意：此操作将立即重启系统，请确保已保存所有工作。

    参数:
        confirm: 必须设置为 True 才能执行重启，默认 False。
        delay: 延迟重启的秒数（仅 Windows 支持；Linux 下忽略，默认为 0）
    返回:
        操作结果或错误信息
    """

    if not confirm:
        return "操作已取消：必须设置 confirm=True 才能执行重启。"

    system = platform.system().lower()
    try:
        if system == "windows":
            # Windows 使用 shutdown 命令
            cmd = ["shutdown", "/r", "/t", str(delay)]
            # 可选添加 /f 强制关闭应用程序
            # cmd.append("/f")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return f"正在重启系统（延迟 {delay} 秒）..."
        elif system == "linux":
            # Linux 使用 reboot 命令，需要 sudo 权限（或 root）
            # 如果当前用户没有 sudo 权限且无密码，会失败
            cmd = ["sudo", "reboot"]
            if delay > 0:
                # Linux 可以使用 shutdown -r +<minutes>，但为了简单，使用 reboot 忽略 delay
                # 如果希望支持 delay，可以用 shutdown -r +<minutes>
                cmd = ["sudo", "shutdown", "-r", f"+{delay//60}"] if delay > 0 else ["sudo", "reboot"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return f"正在重启系统..."
        elif system == "darwin":  # macOS
            # macOS 同样使用 sudo reboot
            cmd = ["sudo", "reboot"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return f"正在重启系统..."
        else:
            return f"不支持的操作系统：{platform.system()}"
    except subprocess.CalledProcessError as e:
        return f"重启失败：{e.stderr or e.stdout or '未知错误'}"
    except Exception as e:
        return f"执行重启命令时出错：{e}"