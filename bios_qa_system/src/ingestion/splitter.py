# src/ingestion/splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """
    切分文档，但保留以下类型的块不切分：
    - metadata["type"] == "table"
    - metadata["category"] == "Table" (Unstructured 生成)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        keep_separator=False,
    )
    final_chunks = []
    for doc in docs:
        meta = doc.metadata
        if meta.get("type") == "table" or meta.get("category") == "Table":
            final_chunks.append(doc)
        else:
            chunks = text_splitter.split_documents([doc])
            final_chunks.extend(chunks)
    return final_chunks