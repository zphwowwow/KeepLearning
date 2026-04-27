import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# 模型与分词器
model_name = "Qwen/Qwen3-0.6B"

model_dir = snapshot_download("Qwen/Qwen3-0.6B",
                              local_dir="Qwen/Qwen3-0.6B",
                              revision="master")

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
	do_sample=True,
	temperature=0.6,  
	top_p=0.95, 
	top_k=20,
)

# 加载、处理数据集和测试集
train_dataset_path = '../data/train.jsonl'
test_dataset_path = '../data/val.jsonl'

train_ds = Dataset.from_json(train_dataset_path)

# 文本最大长度
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
# 随机打乱训练集，打破数据中的数据相关性，提高模型泛化能力（如果是序列任务则不做，避免丢失数据顺序本身包含的重要信息）
train_dataset = train_dataset.shuffle(seed=42)

# 处理测试数据集，并删除原始的文本
val_ds = Dataset.from_json(test_dataset_path)
val_dataset = val_ds.map(process_func, remove_columns=['input','output','instruction'])

# 训练参数
training_args = TrainingArguments(
    output_dir="./qwen3-0.6B-fft",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=3,
    learning_rate=1e-5,
    warmup_steps=20,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="steps",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    bf16=True if torch.cuda.is_available() else False,
    gradient_checkpointing=True,
    save_total_limit=5,
    report_to="none"
)

# 构造训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

# 开始训练
torch.cuda.empty_cache()  # 释放残留内存,训练前执行
trainer.train()
