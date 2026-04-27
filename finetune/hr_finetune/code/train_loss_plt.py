"""
读取模型微调时保存的训练状态文件，并将训练过程的指标绘图打印
"""

import json
import matplotlib.pyplot as plt
import os
import re

# 获取训练输出目录下最新的checkpoint目录
def get_latest_chpoint(path, prefix="checkpoint-"):
    # 列出所有匹配的文件夹，并提取数字后缀
    folders = [
        (int(re.search(rf"{re.escape(prefix)}(\d+)$", name).group(1)), name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name)) and re.match(rf"{re.escape(prefix)}\d+$", name)
    ]
    # 如果没有符合条件的文件夹，返回 None
    if not folders:
        return None
    # max 按数字后缀排序，取最大
    return path + "/" + max(folders)[1]

# 读取训练状态文件
file = get_latest_chpoint('./output')+ '/trainer_state.json'
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取训练日志
log_history = data['log_history']

# 分离训练和验证数据
train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
train_steps = [entry['step'] for entry in log_history if 'loss' in entry]

eval_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
eval_steps = [entry['step'] for entry in log_history if 'eval_loss' in entry]

# 绘制双曲线
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, 'b-', label='Train Loss')
plt.plot(eval_steps, eval_loss, 'orange', marker='o', linestyle='-', label='Eval Loss')

# 图表装饰
plt.title('Training & Validation Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 显示图表
plt.show()
