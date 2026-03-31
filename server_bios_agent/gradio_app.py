import gradio as gr
import requests
import uuid
from typing import Dict

# 本地后端地址（启动 uvicorn 后默认端口 8000）
BACKEND_URL = "http://localhost:8000/chat"

sessions: Dict[str, str] = {}

def get_or_create_session(username: str) -> str:
    if username not in sessions:
        sessions[username] = str(uuid.uuid4())
    return sessions[username]

def chat(message, history, username):
    session_id = get_or_create_session(username)
    try:
        resp = requests.post(BACKEND_URL, json={
            "session_id": session_id,
            "message": message
        }, timeout=120)
        if resp.status_code == 200:
            answer = resp.json()["answer"]
            return answer
        else:
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="Server BIOS AI Agent") as demo:
    gr.Markdown("# Server BIOS AI Agent")
    gr.Markdown("自然语言控制服务器 BIOS 调试、测试、诊断。")
    
    username = gr.Textbox(label="Your Name (for session)", value="default")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    clear = gr.Button("Clear")
    
    def respond(message, chat_history, username):
        bot_message = chat(message, chat_history, username)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot, username], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860)