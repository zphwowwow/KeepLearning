from fastmcp import FastMCP

# 导入各工具函数
from tools.basic import introduce
from tools.apps import open_neteasemusic
from tools.system import restart_computer
from tools.file import read_file, write_file, list_directory, run_script

# 创建 MCP 服务器实例
mcp = FastMCP(name="CamelliaTools")

# 注册工具
mcp.tool(introduce)
mcp.tool(open_neteasemusic)
mcp.tool(restart_computer)
mcp.tool(read_file)
mcp.tool(write_file)
mcp.tool(list_directory)
mcp.tool(run_script)

if __name__ == "__main__":
    # 使用 SSE 传输模式启动
    mcp.run(transport="sse", host="0.0.0.0", port=8080)