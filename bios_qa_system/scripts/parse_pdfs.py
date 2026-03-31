import os
import json
import fitz  # pymupdf
from paddleocr import PaddleOCR
from tqdm import tqdm

def parse_pdf_with_ocr(pdf_path, ocr_engine, output_dir):
    """解析PDF，合并文本和OCR结果"""
    doc = fitz.open(pdf_path)
    doc_name = os.path.basename(pdf_path).replace(".pdf", "")
    all_pages = []
    
    for page_num, page in enumerate(doc):
        # 尝试直接提取文本
        text = page.get_text()
        if len(text.strip()) < 50:  # 文本太少，使用OCR
            # 将页面转为图片
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 放大2倍
            img_path = f"/tmp/page_{page_num}.png"
            pix.save(img_path)
            # OCR识别
            result = ocr_engine.ocr(img_path, cls=True)
            if result and result[0]:
                text = "\n".join([line[1][0] for line in result[0]])
            os.remove(img_path)
        
        all_pages.append({
            "page": page_num,
            "content": text.strip(),
            "source": doc_name
        })
    
    # 保存为JSON
    output_path = os.path.join(output_dir, f"{doc_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)
    return output_path

def main():
    # 初始化OCR（第一次运行会自动下载模型）
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 英文文档
    
    pdf_dir = "./data/pdfs"
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        parse_pdf_with_ocr(pdf_path, ocr, output_dir)

if __name__ == "__main__":
    main()