import gradio as gr
from agent.agent import run_agent

MCP_SERVER_URL = "http://localhost:8001"

def chat(message, history):
    answer = run_agent(MCP_SERVER_URL, message)
    return answer

with gr.Blocks(title="BIOS Debug Assistant") as demo:
    gr.Markdown("# BIOS Debug Assistant")
    gr.Markdown("自然语言控制BIOS调试工具链（基于MCP协议）")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    clear = gr.Button("Clear")
    
    def respond(message, chat_history):
        bot_message = chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860)