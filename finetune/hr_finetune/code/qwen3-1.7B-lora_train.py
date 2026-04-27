import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

"""
本代码为模型微调，步骤：
1. 下载、加载模型
2. 数据格式转换，得到 input_ids（数据转换为Tokens）  attention_mask（全是1） label（标准答案）
3. LoRA 参数设置
4. 微调训练参数设置、执行训练
"""

# 从modelscope上下载Qwen3模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen3-1.7B",
                              local_dir="Qwen/Qwen3-1.7B",
                              revision="master")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                          use_fast=False,  
                                          trust_remote_code=True) 

# 加载预训练模型，可通过参数压缩模型，控制模型输出效果
model = AutoModelForCausalLM.from_pretrained(model_dir,
                                             device_map="auto",
                                             torch_dtype="auto", 
                                             #以下为官方建议的最佳参数
                                             do_sample=True,
                                             temperature=0.6,  
                                             top_p=0.95, 
                                             top_k=20,
                                             )  

# LoRA层的训练需要使用输入张量的梯度计算结果
model.enable_input_require_grads()

# 指定训练集和验证集
train_dataset_path = '../data/train.jsonl'
val_dataset_path = '../data/val.jsonl'

# 加载训练集
train_ds = Dataset.from_json(train_dataset_path)
print("训练集数据：\n",train_ds[0])

# 文本最大长度，一般输入+输出长度在几百字~几千字内的话可设置在1024~2048这个范围内
MAX_LENGTH = 1024

def process_func(example):
    """
    对数据集进行预处理，example为输入数据
    利用分词器将数据处理为模型可接收参数的格式
    返回：input_ids, attention_mask, labels（用于与大模型输出的结果计算损失）
    """
    # 消息构造
    messages = [
        {"role": "system", "content": example["instruction"]}, # 指令
        {"role": "user", "content": example["input"]},  # 用户问题
    ]
    
    text = tokenizer.apply_chat_template(
        messages,  
        tokenize=False,  
        add_generation_prompt=True,  # 是否添加需要模型生成回复的提示词
        enable_thinking=True  # 是否开启思考模式
    )
    
    #将按消息模板构造好的instruction+input通过分词器转化为token
    inputs = tokenizer(text,add_special_tokens=True)
    # 将标准答案转为token
    response = tokenizer(example['output'],add_special_tokens=False)
    
    # input_ids  = 用户输入+标准答案
    input_ids = inputs["input_ids"] + response["input_ids"]
    # attention_mask 通过 0/1 控制有哪些token参与自注意力的计算
    attention_mask = inputs["attention_mask"] + response["attention_mask"]
    
    # labels 用于让模型对比输出与标准答案，从而计算损失
    # 大模型生成结果时以“原始问题+答案”的形式输出，为了保证长度一致
    # 需要在标准答案前添加与原始问题长度相同的多个 -100（-100表示不参与计算）
    labels = ([-100] * len(inputs["input_ids"]) + response["input_ids"])
    
    # 根据设定的文本最大长度，对输入和输出进行截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH-1]
        attention_mask = attention_mask[:MAX_LENGTH-1]
        labels = labels[:MAX_LENGTH-1]
        
    # 统一在末尾加上截断符以及与之对应的mask值
    input_ids = input_ids+[tokenizer.eos_token_id]
    attention_mask = attention_mask+[1]
    labels = labels+[tokenizer.eos_token_id]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 处理训练数据集，并删除原始的文本
train_dataset = train_ds.map(process_func, remove_columns=['input','output','instruction'])
# 随机打乱训练集，打破数据中的数据相关性，提高模型泛化能力（如果是序列任务则不做，避免丢失数据顺序本身包含的重要信息
train_dataset = train_dataset.shuffle(seed=42)
# print("分词器处理后的训练集数据：\n",train_dataset[0])


# # 若没有提取准备测试集，则可以使用如下代码划分测试集用于测试
# train_dataset = train_dataset.train_test_split(test_size=0.2,shuffle=True)
# train_data = train_dataset['train']
# val_data = train_dataset['test']

# 处理验证数据集，并删除原始的文本
val_ds = Dataset.from_json(val_dataset_path)
val_dataset = val_ds.map(process_func, remove_columns=['input','output','instruction'])
# print("测试集数据：\n",val_dataset[0])

## LoRA微调配置
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(task_type=TaskType.CAUSAL_LM, # 任务类型
                    # 是否启用推理模式
                    inference_mode=False,
                    # 低秩矩阵的维数。r通常使用4，8，16，32，64
                    # 值越小模型计算开销越低
                    r=8,
                    # 低秩矩阵的缩放系数，通常设为r的2倍，也可设更大的值（典型是r的2~32倍之间）
                    lora_alpha=16,
                    lora_dropout=0.05, # LoRA层的丢弃率，取值范围为[0, 1)，被选定的神经元对下游神经元激活的贡献在前向传递中被暂时删除，且任何权重更新都不会应用于后向传递的神经元
                    # 不同模型中包含的模块名称不一样，也可以自行指定，最基本的是q_proj、v_proj
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    )
# 将模型包装为微调模型
model = get_peft_model(model, config)

# 训练参数设置
training_args = TrainingArguments(
    output_dir="./output",  # 结果保存目录
    num_train_epochs=20,    # 训练轮数，默认值为3
    eval_strategy="steps",  # 验证方法步骤，可选：epochs, steps，设为step后需对应设置eval_steps，其值默认为500
    eval_steps=50,      # 验证步数，可设整数或[0,1]，<1时将被转化为总的训练步数的比率。该值缺省的时候自动赋logging_steps（默认值500）。
    learning_rate=5e-4,  # 学习率，默认为5e-5
    per_device_train_batch_size=1,    # 每个设备上的训练批大小。
    per_device_eval_batch_size=1,     # 每个设备上的评估批大小。
    warmup_steps=20,    # 预热步数比例
    weight_decay=0.01,  # 权重衰减。
    logging_strategy="steps",  # 日志策略，可选：epochs, steps
    logging_steps=10,  # 日志步数。
    save_steps=50,     # 保存模型步数。
    save_total_limit=10, # 最多保存多个checkpoint文件
    report_to="none",   # 不使用报告
    # gradient_accumulation_steps=3,  # 梯度累积步数。
    # gradient_checkpointing=True,
    gradient_checkpointing=False,  #开启梯度检查点，节省显存，但时间成本会增高
)

# 构建训练器
trainer = Trainer(
    model=model,   # 模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=val_dataset,     # 验证数据集
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 数据处理器，必填
)

# 执行训练
torch.cuda.empty_cache()  # 释放残留内存,训练前执行
trainer.train()

# 模型效果改进
# 1. 数据量增加， 同一个事情，用不同的问题表达。或者用不同模型生成问题和答案
# 多次划分文本块，使用不同的长度等生成文本块
# 2. 模型参数增加， 模型参数增加，4B  8B模型
# 3. 修改微调参数，使得效果更好