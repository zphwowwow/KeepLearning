import chromadb
from chromadb.utils import embedding_functions
from app.config import Config

class RAGRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
        self.collection = self.client.get_collection(
            name="bios_docs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=Config.EMBEDDING_MODEL
            )
        )
    
    def retrieve(self, query, top_k=Config.TOP_K):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        # 整理返回
        contexts = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            contexts.append({
                "content": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": 1 - dist  # 相似度得分
            })
        return contexts