import os
import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

def chunk_text(text, max_tokens=300, overlap=50):
    """简单按段落切分，后续可优化"""
    paragraphs = text.split("\n\n")
    chunks = []
    for p in paragraphs:
        if len(p.strip()) < 20:
            continue
        # 简单按字数切分（实际应用可用tokenizer）
        words = p.split()
        for i in range(0, len(words), max_tokens - overlap):
            chunk = " ".join(words[i:i+max_tokens])
            chunks.append(chunk)
    return chunks

def main():
    # 使用本地的sentence-transformers模型作为embedding函数
    # 这里使用Chroma自带的函数，但需要指定模型
    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(
        name="bios_docs",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-zh-v1.5"
        )
    )
    
    # 加载所有已处理的文档
    processed_dir = "./data/processed"
    json_files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for idx, json_file in enumerate(tqdm(json_files, desc="Processing chunks")):
        with open(os.path.join(processed_dir, json_file), "r", encoding="utf-8") as f:
            pages = json.load(f)
        for page in pages:
            content = page["content"]
            if not content:
                continue
            chunks = chunk_text(content)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_idx": chunk_idx
                })
                all_ids.append(f"{page['source']}_p{page['page']}_c{chunk_idx}")
    
    # 批量插入
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            ids=all_ids[i:i+batch_size],
            documents=all_chunks[i:i+batch_size],
            metadatas=all_metadatas[i:i+batch_size]
        )
    
    print(f"Inserted {len(all_chunks)} chunks into vector DB.")

if __name__ == "__main__":
    main()