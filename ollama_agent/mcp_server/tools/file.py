import subprocess
import shlex
from pathlib import Path

# ========== 配置区域 ==========
# 工作区根目录（所有文件操作限制在此目录内）
# WORKSPACE = Path.home() /
WORKSPACE = Path.cwd()
WORKSPACE.mkdir(exist_ok=True)

# 脚本存放目录（必须位于 WORKSPACE 下）
SCRIPTS_DIR = WORKSPACE / "scripts"
SCRIPTS_DIR.mkdir(exist_ok=True)

# 脚本白名单（为空表示允许所有位于 SCRIPTS_DIR 内的脚本，但推荐填写具体文件名）
ALLOWED_SCRIPTS = {"hello.py"}   # 例如允许这些脚本

# ========== 辅助函数 ==========
def safe_path(user_path: str) -> Path:
    """
    将用户输入的路径转换为安全路径，确保在 WORKSPACE 内。
    如果路径试图逃逸 WORKSPACE，则抛出 ValueError。
    """
    requested_path = (WORKSPACE / user_path).resolve()
    if not requested_path.is_relative_to(WORKSPACE):
        raise ValueError(f"路径 '{user_path}' 超出了允许的工作区 {WORKSPACE}")
    return requested_path

def safe_script_path(script_name: str) -> Path:
    """
    确保脚本位于 SCRIPTS_DIR 内，并检查白名单。
    """
    script_path = (SCRIPTS_DIR / script_name).resolve()
    if not script_path.is_relative_to(SCRIPTS_DIR):
        raise ValueError(f"脚本 '{script_name}' 不在允许的脚本目录 {SCRIPTS_DIR} 内")
    if ALLOWED_SCRIPTS and script_name not in ALLOWED_SCRIPTS:
        raise ValueError(f"脚本 '{script_name}' 不在可执行白名单中。允许的脚本：{', '.join(ALLOWED_SCRIPTS)}")
    if not script_path.exists() or not script_path.is_file():
        raise FileNotFoundError(f"脚本文件 '{script_name}' 不存在")
    return script_path

def read_file(file_path: str) -> str:
    """
    读取指定文件的内容。
    参数:
        file_path: 相对于工作区的文件路径，例如 "notes.txt" 或 "subdir/data.json"
    返回:
        文件内容文本，如果文件不存在则返回错误信息。
    """
    try:
        path = safe_path(file_path)
        if not path.exists():
            return f"错误：文件 '{file_path}' 不存在。"
        if path.is_dir():
            return f"错误：'{file_path}' 是一个目录，不是文件。"
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"读取文件时出错: {e}"

def write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    将内容写入文件（如果文件已存在且 overwrite=False 则拒绝）。
    参数:
        file_path: 相对于工作区的文件路径
        content: 要写入的文本内容
        overwrite: 是否覆盖已存在的文件，默认为 False
    返回:
        操作结果消息
    """
    try:
        path = safe_path(file_path)
        if path.exists() and not overwrite:
            return f"错误：文件 '{file_path}' 已存在。如需覆盖请设置 overwrite=True。"
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件 '{file_path}'"
    except Exception as e:
        return f"写入文件时出错: {e}"

def list_directory(dir_path: str = "") -> str:
    """
    列出指定目录下的文件和子目录。
    参数:
        dir_path: 相对于工作区的目录路径，默认为工作区根目录
    返回:
        目录内容列表，每行一个项目（目录以 '/' 结尾）
    """
    try:
        path = safe_path(dir_path) if dir_path else WORKSPACE
        if not path.exists():
            return f"错误：目录 '{dir_path or '.'}' 不存在。"
        if not path.is_dir():
            return f"错误：'{dir_path or '.'}' 不是一个目录。"
        items = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"{item.name}/")
            else:
                items.append(item.name)
        if not items:
            return "目录为空。"
        return "\n".join(items)
    except Exception as e:
        return f"列出目录时出错: {e}"

def run_script(script_name: str, args: str = "") -> str:
    """
    执行 scripts 目录下的指定脚本（支持 .py, .sh, .bat/.cmd）。
    参数:
        script_name: 脚本文件名（例如 "hello.py"）
        args: 传递给脚本的命令行参数（可选，用空格分隔，支持引号）
    返回:
        脚本的标准输出和错误信息
    """
    try:
        script_path = safe_script_path(script_name)

        # 安全解析参数（支持带空格的引号参数）
        args_list = shlex.split(args) if args else []

        # 根据扩展名构建命令
        suffix = script_path.suffix.lower()
        if suffix == '.py':
            cmd = ['python', str(script_path)] + args_list
        elif suffix == '.sh':
            cmd = ['bash', str(script_path)] + args_list
        elif suffix in ('.bat', '.cmd'):
            cmd = ['cmd', '/c', str(script_path)] + args_list
        else:
            # 默认尝试直接执行（需要可执行权限）
            cmd = [str(script_path)] + args_list

        # 执行命令，超时 30 秒
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=SCRIPTS_DIR
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[返回码: {result.returncode}]"
        return output if output else "脚本执行完成，无输出。"
    except FileNotFoundError as e:
        return f"脚本文件或解释器未找到：{e}"
    except subprocess.TimeoutExpired:
        return "错误：脚本执行超时（30秒）。"
    except Exception as e:
        return f"执行脚本时出错: {e}"

