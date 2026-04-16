#!/usr/bin/env python3
# scripts/build_knowledge_base.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pipeline import build_knowledge_base
import json

if __name__ == "__main__":
    chunks = build_knowledge_base(raw_dir="data/raw", chunk_size=800, chunk_overlap=100)
    # 保存为 JSON 以供后续使用
    output_path = "data/processed/chunks.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(chunks)} 个 chunks 到 {output_path}")