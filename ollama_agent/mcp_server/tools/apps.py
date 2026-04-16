import platform
import subprocess
import shutil
import os

def open_neteasemusic(app_path: str = "") -> str:
    """打开网易云音乐客户端"""
    system = platform.system().lower()

    if app_path:
        path = app_path
    else:
        if system == "windows":
            candidates = [
                r"C:\Program Files (x86)\Netease\CloudMusic\cloudmusic.exe",
                r"C:\Program Files\Netease\CloudMusic\cloudmusic.exe",
                r"%LOCALAPPDATA%\Netease\CloudMusic\cloudmusic.exe",
                r"D:\LenovoSoftstore\Install\wangyiyunyinle\cloudmusic.exe"
            ]
            candidates = [os.path.expandvars(p) for p in candidates]
            path = None
            for cand in candidates:
                if os.path.exists(cand):
                    path = cand
                    break
            if not path:
                path = shutil.which("cloudmusic.exe")
        elif system == "darwin":
            path = "/Applications/NetEaseMusic.app"
            if not os.path.exists(path):
                user_app = os.path.expanduser("~/Applications/NetEaseMusic.app")
                if os.path.exists(user_app):
                    path = user_app
                else:
                    path = "NetEaseMusic"
        elif system == "linux":
            path = "netease-cloud-music"
            if not shutil.which(path):
                path = None
        else:
            return f"不支持的操作系统: {platform.system()}"

    if not path:
        return "未找到网易云音乐，请提供正确的 app_path 参数。"

    try:
        if system == "windows":
            subprocess.Popen([path], shell=False)
        elif system == "darwin":
            if path.endswith(".app") and os.path.isdir(path):
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen([path])
        else:
            subprocess.Popen([path])
        return f"已启动网易云音乐 (路径: {path})"
    except Exception as e:
        return f"启动失败: {e}"