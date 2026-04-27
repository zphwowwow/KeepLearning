import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'E:/enviroment/HuggingFaceCache'
import sys
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

model_name = "unsloth/Qwen2.5-1.5B-Instruct"
lora_path = os.path.join(PROJECT_ROOT, "outputs/lora")

# 加载基座模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    load_in_4bit=True,
)

# 加载 LoRA 权重
if os.path.exists(lora_path):
    model = PeftModel.from_pretrained(model, lora_path)
    print("已加载微调后的 LoRA 权重")
else:
    print(f"警告: 未找到 LoRA 权重 {lora_path}，将使用原始模型")

FastLanguageModel.for_inference(model)

print("BIOS 问答模型已加载。输入 'quit' 退出。")
while True:
    user = input("你: ")
    if user.lower() == 'quit':
        break
    messages = [{"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"助手: {response}\n")