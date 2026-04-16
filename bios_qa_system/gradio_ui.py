import gradio as gr
import requests

API_URL = "http://localhost:8000/query"


def ask_question(question: str) -> str:
    """调用 FastAPI 接口，返回答案文本（含来源）"""
    if not question.strip():
        return "请输入有效问题。"
    payload = {"question": question, "top_k": 3}
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        answer = data["answer"]
        sources = data["sources"]
        if sources:
            source_text = "\n\n**📖 参考来源：**\n" + "\n".join(
                f"- {src['source_id']} (第 {src['page']} 页)" for src in sources
            )
            return answer + source_text
        else:
            return answer + "\n\n（未找到相关来源）"
    except Exception as e:
        return f"❌ 请求失败：{str(e)}"


def respond(message: str, chat_history: list):
    """处理用户消息，返回清空输入框和更新后的对话历史（字典列表格式）"""
    if not message:
        return "", chat_history
    answer = ask_question(message)
    # 使用字典格式追加消息
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history


with gr.Blocks(title="BIOS 知识库问答", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔧 BIOS 技术问答助手")
    gr.Markdown("基于本地知识库 + Ollama 本地模型")

    # 不指定 type 参数，让 Gradio 使用默认格式（新版默认字典）
    chatbot = gr.Chatbot(label="对话历史", height=500)
    msg = gr.Textbox(label="输入您的问题", placeholder="例如：0xE7 报错怎么解决？")
    clear = gr.Button("清空对话")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)  # 清空历史

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)