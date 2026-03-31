import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Ollama 配置
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")
    
    # Redis 配置
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Agent 配置
    MAX_AGENT_LOOPS = int(os.getenv("MAX_AGENT_LOOPS", "10"))
    
    # 工具调用方式：True=原生Tool Call, False=手动ReAct文本解析
    USE_NATIVE_TOOL_CALL = os.getenv("USE_NATIVE_TOOL_CALL", "true").lower() == "true"