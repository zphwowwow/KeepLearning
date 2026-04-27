def introduce() -> str:
    """介绍本服务器的所有功能"""
    descriptions = [
        "• greet(name): 向指定的人打招呼",
        "• add(a, b): 计算两个整数的和",
        "• read_file(file_path): 读取工作区内的文件内容",
        "• write_file(file_path, content, overwrite=False): 写入文件内容到工作区",
        "• list_directory(dir_path=''): 列出工作区目录下的文件和子目录",
        "• run_script(script_name, args=''): 执行 scripts 目录下的指定脚本",
        "• restart_computer(confirm=False, delay=0): 重启计算机（需 confirm=True）",
        "• open_neteasemusic(app_path=''): 打开网易云音乐客户端",
        "• get_weather(city): 获取指定城市的实时天气",
        "• introduce(): 查看本服务器的功能介绍",
    ]
    intro = "我是多功能助手，提供以下工具：\n\n" + "\n".join(descriptions)
    intro += "\n\n你可以直接提出需求，我会自动调用合适的工具。如果需要了解某个工具的详细用法，可以询问我。"
    return intro
