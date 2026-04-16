# scripts/load_to_vector_store.py
import sys
import os
import json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'E:/enviroment/HuggingFaceCache'

# 将项目根目录加入 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from langchain_core.documents import Document
from src.retrieval.vector_store import ChromaStore

def load_chunks_from_json(json_path: str):
    """从 JSON 文件加载 chunks，返回 Document 列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        doc = Document(page_content=item["page_content"], metadata=item["metadata"])
        docs.append(doc)
    print(f"从 {json_path} 加载了 {len(docs)} 个文档块")
    return docs

if __name__ == "__main__":
    json_path = "data/processed/chunks.json"
    if not os.path.exists(json_path):
        print(f"错误: {json_path} 不存在，请先运行 build_knowledge_base.py 生成该文件")
        sys.exit(1)

    # 1. 加载 chunks
    chunks = load_chunks_from_json(json_path)

    # 2. 初始化 ChromaStore（持久化目录）
    store = ChromaStore(persist_directory="data/vector_store")

    # 3. 添加到集合（如果集合已存在，会先删除旧集合？这里选择清空重建）
    collection_name = "bios_kb"
    try:
        store.client.delete_collection(collection_name)
        print(f"已删除旧集合 {collection_name}")
    except:
        pass

    # 添加文档
    store.add_documents(chunks, collection_name=collection_name)
    print(f"成功将 {len(chunks)} 个文档块存入 Chroma 集合 {collection_name}")

    # 4. 简单测试检索
    query = "BIOS 与 CMOS 有什么区别？"
    results = store.search(query, top_k=3, collection_name=collection_name)
    print(f"\n检索测试: {query}")
    for i, res in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"  内容: {res['text'][:200]}...")
        print(f"  元数据: {res['metadata']}")
        print(f"  距离: {res['distance']}")