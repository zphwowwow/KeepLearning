from pathlib import Path
from openai import OpenAI

key = Path("./modelscope-key.txt").read_text()

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=key, 
)

response = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-V3.1', 
    messages=[
        {
            'role': 'user',
            'content': '公司管理层早会时间和地点'
        }
    ],
    stream=False
)

print(response.choices[0].message.reasoning_content)
print('\n\n === Final Answer ===\n')
print(response.choices[0].message.content)