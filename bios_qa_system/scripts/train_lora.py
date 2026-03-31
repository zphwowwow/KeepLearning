import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json

# 准备数据：需要构造指令格式
def prepare_data(data_path):
    """
    数据格式示例：每行一个JSON，包含 "instruction", "input", "output"
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    prompts = []
    for item in data:
        prompt = f"<|im_start|>user\n{item['instruction']}\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        prompts.append(prompt)
    
    return Dataset.from_dict({"text": prompts})

def train():
    model_name = "Qwen/Qwen-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # 数据加载
    dataset = prepare_data("./data/train.jsonl")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./models/qwen-7b-chat-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        save_steps=500,
        logging_steps=50,
        remove_unused_columns=False,
    )
    
    # 自定义data_collator
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["input_ids"]
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    
    trainer.train()
    model.save_pretrained("./models/qwen-7b-chat-lora")
    tokenizer.save_pretrained("./models/qwen-7b-chat-lora")

if __name__ == "__main__":
    train()