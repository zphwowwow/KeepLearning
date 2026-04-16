# src/ingestion/loader.py
import os
import json
import csv
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document

from src.ingestion.parser import extract_tables_from_pdf, is_scanned_pdf, ocr_pdf

# ========== PDF 加载 ==========
def load_pdf_auto(file_path: str) -> List[Document]:
    """自动判断 PDF 类型：扫描件 -> OCR，普通文本型 -> PyPDFLoader（可选同时提取表格）"""
    if is_scanned_pdf(file_path):
        print(f"  扫描件 PDF: {os.path.basename(file_path)}，使用 OCR")
        return ocr_pdf(file_path)
    else:
        print(f"  普通 PDF: {os.path.basename(file_path)}，使用 PyPDFLoader + 表格提取")
        # 同时提取表格和普通文本
        docs = []
        docs.extend(extract_tables_from_pdf(file_path))
        docs.extend(PyPDFLoader(file_path).load())
        return docs

# ========== Word 加载 ==========
def load_word(file_path: str) -> List[Document]:
    print(f"  Word 文档: {os.path.basename(file_path)}，使用 UnstructuredWordDocumentLoader")
    loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    return loader.load()

# ========== Excel 加载 ==========
def load_excel(file_path: str) -> List[Document]:
    print(f"  Excel 表格: {os.path.basename(file_path)}，使用 UnstructuredExcelLoader")
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    return loader.load()

# ========== Jira JSON 加载 ==========
def load_jira_json(file_path: str) -> List[Document]:
    print(f"  Jira JSON: {os.path.basename(file_path)}，正在解析工单")
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    issues = data.get("issues", []) if isinstance(data, dict) else data
    for issue in issues:
        text = f"标题: {issue.get('summary', '')}\n描述: {issue.get('description', '')}\n"
        comments = issue.get("comments", [])
        if comments:
            text += "评论:\n" + "\n".join(f"  - {c}" for c in comments)
        resolution = issue.get("resolution")
        if resolution:
            text += f"\n解决方案: {resolution}"
        metadata = {
            "source_type": "jira",
            "source_id": issue.get("key"),
            "status": issue.get("status"),
            "created": issue.get("created"),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def load_jira_csv(file_path: str) -> List[Document]:
    print(f"  Jira CSV: {os.path.basename(file_path)}，正在解析工单")
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = f"标题: {row.get('summary', '')}\n描述: {row.get('description', '')}\n"
            comments = row.get("comments", "")
            if comments:
                text += f"评论: {comments}\n"
            resolution = row.get("resolution", "")
            if resolution:
                text += f"解决方案: {resolution}"
            metadata = {
                "source_type": "jira",
                "source_id": row.get("key"),
                "status": row.get("status"),
            }
            docs.append(Document(page_content=text, metadata=metadata))
    return docs

# ========== 统一入口 ==========
def load_document(file_path: str) -> List[Document]:
    """根据文件扩展名自动选择合适的加载器"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf_auto(file_path)
    elif ext in [".docx", ".doc"]:
        return load_word(file_path)
    elif ext in [".xlsx", ".xls"]:
        return load_excel(file_path)
    elif ext == ".json":
        return load_jira_json(file_path)
    elif ext == ".csv":
        return load_jira_csv(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")