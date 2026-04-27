# 多线程提交评估请求给不同高阶模型
import asyncio
import openai
from openai import AsyncOpenAI
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 初始化调用高阶大模型的Client
client_mdscope = AsyncOpenAI(
	base_url='https://api-inference.modelscope.cn/v1',
    api_key=Path("./modelscope-key.txt").read_text() 
	) 
MODEL_MDSCOPE = "Qwen/Qwen3-235B-A22B-Instruct-2507"

client_pony = AsyncOpenAI(
	base_url='https://api.tokenpony.cn/v1',
    api_key=Path("./tokenpony-key.txt").read_text() 
	) 
MODEL_PONY = "deepseek-r1-0528"

client_volcen = AsyncOpenAI(
	base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=Path("./volcengine-key.txt").read_text() 
	) 
MODEL_VOLCEN = "ep-20251017152145-k6vct"  #doubao-1-5-thinking-pro

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

basemodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                             torch_dtype="auto",
                                             device_map="auto").eval() #切换到推理模式而非训练模式

loramodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-LoRA",
                                                 device_map="auto",
                                                 torch_dtype="auto").eval() # 切换到推理模式而非训练模式

test_file = "../data/test0.jsonl"  # 测试集文件
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

prompt_template = """
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
"""
questions = []
references = []
lora_answers = []
base_answers = []
for i in test_data:
    questions += [i["instruction"] + i["input"]]
    references += [i["output"].rsplit("</think>\n")[1]]  #只截取回答部分
    lora_answer = generate_answer(i["instruction"] + i["input"],loramodel)
    base_answer = generate_answer(i["instruction"] + i["input"],basemodel)
    lora_answers += [lora_answer]
    base_answers += [base_answer]
    
# 异步请求函数（带重试与退避）
async def ask_model(client, model_name, prompt, query, max_retries=5, retry_interval=30):
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            content = response.choices[0].message.content.strip("\n")
            return model_name, query, content

        except openai.RateLimitError as e:
            print(f"[{model_name}] 速率限制: {e}，{retry_interval}s后重试...")
        except asyncio.TimeoutError:
            print(f"[{model_name}] 超时，{retry_interval}s后重试...")
        except Exception as e:
            print(f"[{model_name}] 其他错误: {e}，{retry_interval}s后重试...")

        await asyncio.sleep(retry_interval)
        retry_interval += 10

    return model_name, query, "API请求失败"

# 异步任务
async def run_async():
    tasks = []
    for i in range(len(test_data)):
        prompt = prompt_template.format(
        ques = questions[i],
        ref = references[i],
        base_answer = base_answers[i],
        lora_answer = lora_answers[i])
        tasks.extend([
            ask_model(client_mdscope, MODEL_MDSCOPE, prompt, questions[i]),
            ask_model(client_pony, MODEL_PONY, prompt, questions[i]),
            ask_model(client_volcen, MODEL_VOLCEN, prompt, questions[i]),
        ])
		
    results = await asyncio.gather(*tasks)
    return results
    
# 运行异步任务
from IPython import get_ipython
loop = asyncio.get_event_loop()

if loop.is_running():
    results = await run_async()
else:
    results = asyncio.run(run_async())
    
# for model_name, query, answer in results:
#     print(f"\n【模型】{model_name}\n【问题】{query}\n【回答】{answer}\n{'-'*60}")

# 综合两个高阶模型的打分给出最终得分
import json
from collections import Counter

scores1 = {"base_score":0,"lora_score":0}
scores2 = {"base_score":0,"lora_score":0}
scores3 = {"base_score":0,"lora_score":0}
for model_name, query, answer in results:
    score = json.loads(answer)
    if(model_name == MODEL_MDSCOPE):
        scores1 = Counter(score)+Counter(scores1)
    elif(model_name == MODEL_PONY):
        scores2 = Counter(score)+Counter(scores2)
    else:
        scores3 = Counter(score)+Counter(scores3)
        
print("高阶模型< {} >给出的评分如下：".format(MODEL_MDSCOPE))
print("基座模型：",scores1["base_score"]/len(test_data))
print("LoRA微调后模型：",scores1["lora_score"]/len(test_data))
print("="*30)
print("高阶模型< {} >给出的评分如下：".format(MODEL_PONY))
print("基座模型：",scores2["base_score"]/len(test_data))
print("LoRA微调后模型：",scores2["lora_score"]/len(test_data))
print("="*30)
print("高阶模型< {} >给出的评分如下：".format(MODEL_VOLCEN))
print("基座模型：",scores3["base_score"]/len(test_data))
print("LoRA微调后模型：",scores3["lora_score"]/len(test_data))
print("="*30)
print("基座模型综合得分：",round((Counter(scores1)+Counter(scores2)+Counter(scores3))["base_score"]/(3*len(test_data)),2))
print("LoRA微调后模型综合得分：",round((Counter(scores1)+Counter(scores2)+Counter(scores3))["lora_score"]/(3*len(test_data)),2))

for i in range(len(test_data)):
    print("="*15,"问题","="*15)
    print(questions[i])
    print("="*14,"标准答案","="*14)
    print(references[i])
    print("="*14,"基座回答","="*14)
    print(base_answers[i])
    print("="*14,"LoRA回答","="*14)
    print(lora_answers[i])