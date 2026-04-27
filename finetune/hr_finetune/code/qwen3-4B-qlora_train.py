'''
注意！量化必须要GPU环境才能跑！！
'''

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import torch
import bitsandbytes as bnb
from modelscope import snapshot_download

model_dir = snapshot_download("Qwen/Qwen3-4B",
                              local_dir="Qwen/Qwen3-4B",
                              revision="master")

# 加载模型和 tokenizer（启用4bit加载）
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    load_in_4bit=True,  # QLoRA主要区别：量化，简单写法，默认nf4
    # load_in_8bit=True,
    device_map="auto",
    torch_dtype="auto",
    do_sample=True,
    temperature=0.6,  
    top_p=0.95, 
    top_k=20,
)

"""
# 使用BitsAndBytesConfig灵活定义
from transformers import BitsAndBytesConfig

# 4位量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  #量化类型
    bnb_4bit_compute_dtype="bfloat16",  #以4位加载和存储模型，在需要时对其进行部分量化，并以16位精度(bfloat16)进行所有计算
    bnb_4bit_use_double_quant=True  #采用QLoRa提出的双量化
)

# 8位量化
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,     # 控制分块阈值，越小越保守(基于每一层（或每个权重矩阵）的激活值判断是否量化
    llm_int8_has_fp16_weight=False  # 是否保留FP16权重副本，更准但更耗显存
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype="auto",
    do_sample=True,
    temperature=0.6,  
    top_p=0.95, 
    top_k=20,
)
"""

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

# 准备模型结构以支持 LoRA 注入，因为模型经过量化，但部分层需要保持高精度
model = prepare_model_for_kbit_training(model)

# 加载、处理数据集和测试集
train_dataset_path = '../data/train.jsonl'
test_dataset_path = '../data/val.jsonl'

# 得到训练集
train_ds = Dataset.from_json(train_dataset_path)

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

# 处理测试数据集，并删除原始的文本
val_ds = Dataset.from_json(test_dataset_path)
val_dataset = val_ds.map(process_func, remove_columns=['input','output','instruction'])

# 配置 LoRA 参数
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=20,
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=5e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=20,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=10,
    logging_strategy="steps",
    logging_steps=10,
    bf16=True,  # 如果你用 fp16 则改成 fp16=True。Qwen3-4B是bf16
    optim="paged_adamw_8bit",  #使用量化后的优化器，更省显存，默认是adam8
    gradient_checkpointing=True,  # 启用梯度检查点
    gradient_checkpointing_kwargs={"use_reentrant": False},  # 显式指定
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset = val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 数据处理器
)

# 微调模型
trainer.train()