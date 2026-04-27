# rag_with_ollama.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'E:/enviroment/HuggingFaceCache'
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval.vector_store import ChromaStore


def ask_question(question, collection_name="bios_kb", top_k=3):
    # 1. 连接向量数据库并检索
    print("正在检索相关知识...")
    store = ChromaStore(persist_directory="data/vector_store")
    results = store.search(question, top_k=top_k, collection_name=collection_name)

    # 2. 从检索结果中提取上下文
    if not results:
        context = "未找到相关文档。"
    else:
        context_parts = []
        for i, result in enumerate(results):
            # 这里可以更灵活地拼接来源信息
            source = result['metadata'].get('source_id', '未知来源')
            page = result['metadata'].get('page', '未知页')
            content = result['text']
            context_parts.append(f"[来源：{source} 第{page}页]\n{content}")
        context = "\n\n".join(context_parts)

    # 3. 调用 Ollama 模型生成回答
    # 初始化 Ollama 模型，默认 Ollama 服务运行在 http://localhost:11434[reference:3]
    llm = ChatOllama(model="qwen2.5:7b")

    # 设计 Prompt 模板，将检索到的知识作为上下文
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的 BIOS 技术问答助手。请仅根据以下提供的参考信息回答用户的问题。"
                   "如果参考信息不足以回答问题，请坦诚告知用户，不要编造答案。回答时，请尽量使用中文。"),
        ("user", "参考信息如下：\n{context}\n\n我的问题是：{question}")
    ])
    # 将 Prompt 模板、检索器和 LLM 串联起来
    chain = prompt_template | llm

    print("正在生成回答...")
    response = chain.invoke({"context": context, "question": question})

    return response.content, results


if __name__ == "__main__":
    # 这是一个简单的测试入口
    # 在实际项目中，你可能想要构建一个更友好的循环或 Web 界面
    while True:
        user_question = input("\n请输入你的问题 (输入 'exit' 退出): ")
        if user_question.lower() == 'exit':
            break
        answer, sources = ask_question(user_question)
        print("\n--- 回答 ---")
        print(answer)
        print("\n--- 信息来源 ---")
        for i, source in enumerate(sources):
            print(f"{i + 1}. 来自 {source['metadata'].get('source_id', '未知')}，"
                  f"第 {source['metadata'].get('page', '未知')} 页")