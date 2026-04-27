# src/ingestion/pipeline.py
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document

from src.ingestion.loader import load_document
from src.ingestion.splitter import split_documents
from src.ingestion.metadata import enrich_metadata

def find_files(root_dir: str, extensions: List[str] = None) -> List[str]:
    """递归查找指定扩展名的所有文件"""
    if extensions is None:
        extensions = [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".json", ".csv"]
    root = Path(root_dir)
    files = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return [str(p) for p in files]

def process_single_file(file_path: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """处理单个文件：加载 → 切分 → 元数据注入"""
    print(f"\n处理文件: {file_path}")
    docs = load_document(file_path)
    print(f"  加载完成，共 {len(docs)} 个原始 Document")
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    print(f"  切分后共 {len(chunks)} 个块")
    chunks = enrich_metadata(chunks, file_path)
    return chunks

def build_knowledge_base(raw_dir: str = "data/raw", chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """扫描 raw_dir 下所有支持的文件，返回所有 chunks"""
    all_files = find_files(raw_dir)
    if not all_files:
        print(f"在 {raw_dir} 目录下未找到任何支持的文件")
        return []
    print(f"找到 {len(all_files)} 个文件")
    all_chunks = []
    for file_path in all_files:
        chunks = process_single_file(file_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    print(f"\n总计生成 {len(all_chunks)} 个文本块")
    return all_chunks