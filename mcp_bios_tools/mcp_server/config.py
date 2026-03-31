import os

class Config:
    MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))