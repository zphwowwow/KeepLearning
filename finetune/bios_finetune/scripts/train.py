import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'E:/enviroment/HuggingFaceCache'
import sys
import yaml
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---------- 设置项目根目录 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------- 环境变量 ----------
#   # 改成你的缓存路径

# ---------- 加载配置 ----------
config_path = os.path.join(PROJECT_ROOT, "config", "train_config.yaml")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"配置文件不存在: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]
lora_cfg = config["lora"]
train_cfg = config["training"]
data_cfg = config["data"]

# ========== 关键修复：强制转换数值类型 ==========
model_cfg["max_seq_length"] = int(model_cfg["max_seq_length"])
model_cfg["load_in_4bit"] = bool(model_cfg["load_in_4bit"])

lora_cfg["r"] = int(lora_cfg["r"])
lora_cfg["alpha"] = int(lora_cfg["alpha"])
lora_cfg["dropout"] = float(lora_cfg["dropout"])

train_cfg["batch_size"] = int(train_cfg["batch_size"])
train_cfg["gradient_accumulation_steps"] = int(train_cfg["gradient_accumulation_steps"])
train_cfg["learning_rate"] = float(train_cfg["learning_rate"])   # 核心修复
train_cfg["num_epochs"] = int(train_cfg["num_epochs"])
train_cfg["logging_steps"] = int(train_cfg["logging_steps"])
train_cfg["save_steps"] = int(train_cfg["save_steps"])
train_cfg["seed"] = int(train_cfg["seed"])
train_cfg["fp16"] = bool(train_cfg["fp16"])
train_cfg["bf16"] = bool(train_cfg["bf16"])
train_cfg["gradient_checkpointing"] = bool(train_cfg["gradient_checkpointing"])

# 数据文件绝对路径
data_file = os.path.join(PROJECT_ROOT, data_cfg["train_file"])
if not os.path.exists(data_file):
    raise FileNotFoundError(f"数据文件不存在: {data_file}")

# ---------- 加载模型 ----------
print(f"正在加载模型: {model_cfg['name']}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_cfg["name"],
    max_seq_length=model_cfg["max_seq_length"],
    load_in_4bit=model_cfg["load_in_4bit"],
    dtype=None,
)

# ---------- 添加 LoRA ----------
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_cfg["r"],
    target_modules=lora_cfg["target_modules"],
    lora_alpha=lora_cfg["alpha"],
    lora_dropout=lora_cfg["dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=train_cfg["seed"],
)

# ---------- 加载数据 ----------
dataset = load_dataset('json', data_files=data_file, split='train')

def format_instruction(example):
    text = f"<s>Human: {example['instruction']}\nAssistant: {example['output']}</s>"
    return {"text": text}

dataset = dataset.map(format_instruction)

# ---------- 训练器 ----------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        max_seq_length=model_cfg["max_seq_length"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_epochs"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        output_dir=os.path.join(PROJECT_ROOT, train_cfg["output_dir"]),
        optim=train_cfg["optim"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        seed=train_cfg["seed"],
    ),
)

# ---------- 开始训练 ----------
trainer.train()

# ---------- 保存模型 ----------
output_dir = os.path.join(PROJECT_ROOT, train_cfg["output_dir"])
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"微调完成，模型保存至 {output_dir}")