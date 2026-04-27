from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

basemodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                             torch_dtype="auto",
                                             device_map="auto").eval() #切换到推理模式而非训练模式

loramodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-LoRA",
                                                 device_map="auto",
                                                 torch_dtype="auto").eval() # 切换到推理模式而非训练模式

test_file = "../data/test.jsonl"  # 测试集文件，格式：[{"input": "...", "output": "..."}]
test_data = Dataset.from_json(test_file)

MAX_NEW_TOKENS = 1024  #限制模型回复长度

# 生成答案函数
def generate_answer(question,mod):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(mod.device)

    with torch.no_grad():
        generated_ids = mod.generate(
            **inputs,
            # max_new_tokens=32768
            max_new_tokens = MAX_NEW_TOKENS
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index],skip_special_tokens=True)
    content = tokenizer.decode(output_ids[index:],skip_special_tokens=True)
    return content.strip("\n")

# 调用modelScope API，使用更高阶的模型判定模型回答是否准确
from openai import OpenAI
from pathlib import Path
import json

api_file = "./modelscope-key.txt"
mastermodel = "Qwen/Qwen3-235B-A22B-Instruct-2507"

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=Path(api_file).read_text() 
    )

def chat_func(client,mastermodel,ques,ref,base_answer,lora_answer):
    response = client.chat.completions.create(
        model = mastermodel, 
        messages=[
            {
                'role': 'user',
                'content':
                f'''
                你是一位专业的大模型评测员。请对比微调前后的模型在回答领域问题时的表现。
                标准答案是基于企业私域知识得出的，因此你在评测时需重点评估两个模型的回答是否能精准覆盖标准答案的关键点。
                【问题】
                {ques}
                【标准答案】
                {ref}
                【base模型的回答】
                {base_answer}
                【lora微调后模型的回答】
                {lora_answer}

                请从大模型的回答对于问题的准确度以及回答覆盖标准答案的完整度出发，
                为base模型的回答以及lora微调后模型的回答打分（满分为100分）：

                请按JSON格式输出分数，注意只需返回单纯的JSON，无需其余文字说明或注释。
                {{
                    "base_score": 基座模型得分,
                    "lora_score": LoRA微调后模型得分
                }}
                '''
            }
        ], # 存疑时可在json中添加一项"reason": 给出以上评分的主要原因 以便找出问题
        stream=False
    )
    
    return response.choices[0].message.content

# 构建一个字典存储高阶大模型输出
from collections import Counter
import re

scores = {
    "base_score": 0,
    "lora_score": 0}
lora_answers = []
base_answers = []

# 遍历测试集每条问答对，将问题分别投入基座模型与LoRA微调模型获取回答
# 将问题、标准答案、基座模型回答、LoRA微调模型回答传入函数，统计得分
try:
    for i in test_data:
        question = i["instruction"] + i["input"]
        reference = i["output"].rsplit("</think>\n")[1]  #只截取回答部分
        lora_answer = generate_answer(question,loramodel)
        base_answer = generate_answer(question,basemodel)
        lora_answers += [lora_answer]
        base_answers += [base_answer]
        
        master_answer = chat_func(client,mastermodel,question,reference,base_answer,lora_answer)

        n_scores = json.loads(master_answer)
        scores = Counter(scores) + Counter(n_scores)
        
except:
    print(master_answer)

print(f"""
综合得分：
    基座模型：{scores["base_score"]/len(test_data)}
    LoRA模型：{scores["lora_score"]/len(test_data)}
""")

# 查看微调前后大模型的回答，主观感受模型效果
for i in range(len(test_data)):
    print("========问题========")
    print(test_data[i]["input"])
    print("========标准答案========")
    print(test_data[i]["output"].rsplit("</think>\n")[1])
    print("========基座回答========")
    print(base_answers[i])
    print("========LoRA回答========")
    print(lora_answers[i])
    print()