import os

class Config:
    # 路径
    DATA_DIR = os.getenv("DATA_DIR", "./data/processed")
    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/qwen-7b-chat-lora")  # 微调后模型路径
    BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen-7B-Chat")  # 基础模型
    
    # RAG参数
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    TOP_K = int(os.getenv("TOP_K", "5"))
    
    # 生成参数
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
    
    # 服务端口
    PORT = int(os.getenv("PORT", "8000"))