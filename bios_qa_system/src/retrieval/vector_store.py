# src/retrieval/vector_store.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class ChromaStore:
    def __init__(self, persist_directory: str = "./data/vector_store", embedding_model: str = "BAAI/bge-m3"):
        self.persist_directory = persist_directory
        # 使用持久化客户端（数据保存在磁盘）
        self.client = chromadb.PersistentClient(path=persist_directory)
        # BGE 嵌入函数
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    
    def get_or_create_collection(self, collection_name: str):
        """获取或创建集合，处理集合不存在的异常"""
        try:
            collection = self.client.get_collection(collection_name)
            print(f"集合 '{collection_name}' 已存在，直接使用。")
        except chromadb.errors.NotFoundError:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            print(f"集合 '{collection_name}' 不存在，已创建。")
        return collection
    
    def add_documents(self, documents: List[Document], collection_name: str = "bios_kb"):
        """将 Document 列表添加到向量库"""
        collection = self.get_or_create_collection(collection_name)
        
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # 生成唯一ID：来源文件_块索引
            source_id = doc.metadata.get("source_id", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", i)
            doc_id = f"{source_id}_{chunk_idx}"
            ids.append(doc_id)
            texts.append(doc.page_content)
            # 过滤掉 None 值，避免 Chroma 报错
            clean_meta = {k: v for k, v in doc.metadata.items() if v is not None}
            metadatas.append(clean_meta)
        
        # 分批添加（每批100条）
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end]
            )
        print(f"成功添加 {len(ids)} 个文档块到集合 '{collection_name}'")
    
    def search(self, query: str, top_k: int = 5, collection_name: str = "bios_kb", 
               filter: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        检索最相似的 top_k 个文档块
        返回列表，每个元素包含 id, text, metadata, distance
        """
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter
        )
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        return retrieved