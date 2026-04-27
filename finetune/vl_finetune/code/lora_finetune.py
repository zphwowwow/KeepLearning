# 依赖库安装命令，终端执行
#pip install uv openai
#uv pip install --system sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
#uv pip install --system --no-deps unsloth_zoo bitsandbytes accelerate "xformers==0.0.32.post2" peft trl triton unsloth
#uv pip install --system transformers==4.57.1
#uv pip install --system --no-deps trl==0.22.2

# 从modelscope上下载Qwen3-vl模型到本地目录下
from modelscope import snapshot_download
model_dir = snapshot_download("Qwen/Qwen3-VL-4B-Instruct",
                              local_dir="Qwen/Qwen3-VL-4B-Instruct",
                              revision="master")
                              
# 模型导入与配置
from unsloth import FastVisionModel 
import torch
import os; os.environ['UNSLOTH_USE_MODELSCOPE'] = '1'

model, tokenizer = FastVisionModel.from_pretrained(
    "/root/autodl-tmp/Qwen/Qwen3-VL-4B-Instruct",
    load_in_4bit = False, 
    use_gradient_checkpointing = "unsloth", # unsloth表示启用智能梯度检查点，自动根据序列长度
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, 
    finetune_language_layers   = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules       = True, 
    
    r = 64,           # 秩越大精度越高，但可能过拟合
    lora_alpha = 128,  # r的倍数，最小=r
    lora_dropout = 0,
    bias = "none",
    use_rslora = False,  
    loftq_config = None, 
    target_modules = "all-linear", # 可通过传入层名列表自定义低秩矩阵的插入位置，当前表示所有线性层都插入低秩矩阵
)

# 数据导入与格式处理
import json
import os; 
os.environ["HF_DATASETS_OFFLINE"] = "1"

with open("data/train_data_v2.jsonl","r",encoding="utf-8") as f:
    train_data = f.readlines()

with open("data/val_data_v2.jsonl","r",encoding="utf-8") as f:
    val_data = f.readlines()

def convert_to_conversations(sample):
    image_id = sample.get("id")
    conversations = []
    for i in sample.get("questions"):
        conversation = [
            { "role": "user",
              "content" : [
                {"type" : "text",  "text"  : i["question"]},
                {"type" : "image", "image" : f"data/images/{image_id}.jpg"} ]
            },
            { "role" : "assistant",
              "content" : [
                {"type" : "text",  "text"  : i["answer"]} ]
            },
        ]
        conversations.append(conversation)
    return conversations
    
json.loads(train_data[0])

train = []
val = []

for i in train_data:
    try:
        train.extend(convert_to_conversations(json.loads(i)))
    except:
        print("异常训练记录：\n",i)
for i in val_data:
    try:
        val.extend(convert_to_conversations(json.loads(i)))
    except:
        print("异常验证记录：\n",i)
        
# 消息模板
"""
[
{ "role": "user",
  "content": [{"type": "text",  "text": Q}, {"type": "image", "image": image} ]
},
{ "role": "assistant",
  "content": [{"type": "text",  "text": A} ]
},
]
"""

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import random

FastVisionModel.for_training(model) # 切换到训练模式

random.shuffle(train)

collator = UnslothVisionDataCollator(model,tokenizer)
collator.image_size=512

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = collator, # 必须指定整理器
    train_dataset = train,
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.03,
        num_train_epochs = 2, 
        learning_rate = 1e-5,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        output_dir = "/root/autodl-tmp/outputs",
        report_to = "none",     # 不指定报告平台

        # 进行视觉微调时必须加入以下参数配置
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

# 确认资源情况
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 开始训练
trainer_stats = trainer.train()

# 模型推理，简单确认效果
FastVisionModel.for_inference(model) # 切换到推理模式

k=0
image = train[k][0].get("content")[1].get("image")
instruction = train[k][0].get("content")[0].get("text")
print(train[k])

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
                   
# 将当前训练好模型保存到指定目录
model.save_pretrained_merged("Qwen/Qwen3-VL-4B-LoRA", tokenizer,)


#或通过checkpoint融合保存
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

#基座模型
model_path = "Qwen/Qwen3-VL-4B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="auto", device_map="auto",do_sample=True,top_p=0.8,top_k=20,temperature=0.7
)
processor = AutoProcessor.from_pretrained(model_path)

# 加载lora权重
latest_chpoint ="outputs/checkpoint-9240-v2"  #名称按照最新的checkpoint目录修改

lora_model = PeftModel.from_pretrained(
    model=model, # 原始模型
    model_id= latest_chpoint # 微调得到的模型快照
)

model_new = lora_model.merge_and_unload()

# 经过模型效果验证后，即可进行模型保存
model_new.save_pretrained("/root/autodl-tmp/Qwen/Qwen3-VL-4B-LoRA")
processor.save_pretrained("/root/autodl-tmp/Qwen/Qwen3-VL-4B-LoRA")
print("模型保存成功~")