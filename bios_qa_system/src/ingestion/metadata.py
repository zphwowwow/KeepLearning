# src/ingestion/metadata.py
import os
from typing import List
from langchain_core.documents import Document

def enrich_metadata(chunks: List[Document], file_path: str, source_type: str = None) -> List[Document]:
    """
    为每个 chunk 添加标准化的元数据
    - 如果 chunk 已有 source_type 元数据（如 jira），则保留；否则使用传入的 source_type 或自动推断
    """
    enriched = []
    base_name = os.path.basename(file_path)
    # 自动推断 source_type（如果未指定且 chunk 没有）
    if source_type is None:
        if "jira" in file_path.lower():
            inferred_type = "jira"
        elif "vendor" in file_path.lower() or "docs" in file_path.lower():
            inferred_type = "vendor_doc"
        else:
            inferred_type = "file"
    else:
        inferred_type = source_type

    # 相对路径（相对于 data/raw）
    if "data/raw" in file_path:
        rel_path = os.path.relpath(file_path, "data/raw")
    else:
        rel_path = file_path

    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata.copy()
        # 如果 chunk 自身没有 source_type，则使用推断值
        if "source_type" not in meta:
            meta["source_type"] = inferred_type
        if "source_id" not in meta:
            meta["source_id"] = base_name
        meta["chunk_index"] = idx
        meta["file_path"] = file_path
        meta["relative_path"] = rel_path
        if "page" not in meta:
            meta["page"] = -1
        # 添加简短标题
        first_line = chunk.page_content.strip().split('\n')[0][:50]
        meta["title"] = first_line if first_line else "无标题"
        enriched.append(Document(page_content=chunk.page_content, metadata=meta))
    return enriched