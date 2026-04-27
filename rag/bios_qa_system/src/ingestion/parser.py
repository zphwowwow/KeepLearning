# src/ingestion/parser.py
import pdfplumber
import pandas as pd
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from typing import List, Optional

# ========== 表格提取 ==========
def extract_tables_from_pdf(file_path: str, pages: Optional[List[int]] = None) -> List[Document]:
    """从PDF中提取所有表格，每个表格生成一个Document（Markdown格式）"""
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if pages and page_num not in pages:
                continue
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                except Exception:
                    df = pd.DataFrame(table)
                table_md = df.to_markdown(index=False)
                metadata = {
                    "page": page_num,
                    "table_index": table_idx,
                    "type": "table",
                    "source": file_path,
                }
                docs.append(Document(page_content=table_md, metadata=metadata))
    return docs

# ========== OCR 扫描件处理 ==========
def is_scanned_pdf(file_path: str, threshold: int = 50) -> bool:
    """判断 PDF 是否为扫描件（通过第一页文字数量）"""
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return True
            text = pdf.pages[0].extract_text() or ""
            return len(text.strip()) < threshold
    except Exception:
        return True

def ocr_pdf(file_path: str, dpi: int = 300, lang: str = "chi_sim+eng") -> List[Document]:
    """使用 pypdfium2 渲染 PDF 页面为图像，然后进行 OCR 识别。无需 Poppler。"""
    docs = []
    pdf = pdfium.PdfDocument(file_path)
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        bitmap = page.render(scale=dpi / 72)
        img = bitmap.to_pil()
        text = pytesseract.image_to_string(img, lang=lang)
        if text.strip():
            metadata = {
                "page": page_num + 1,
                "source": file_path,
                "ocr": True,
                "dpi": dpi,
            }
            docs.append(Document(page_content=text, metadata=metadata))
        else:
            print(f"警告: 第 {page_num + 1} 页 OCR 未识别出任何文本")
    return docs