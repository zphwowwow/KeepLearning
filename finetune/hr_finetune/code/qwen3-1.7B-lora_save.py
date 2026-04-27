"""
本代码为模型合并，将原始模型与微调后的权重参数合并，得到微调后的专家模型
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download

model_dir = snapshot_download("Qwen/Qwen3-1.7B",
                              local_dir="Qwen/Qwen3-1.7B",
                              revision="master")

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", device_map="auto", torch_dtype="auto",
    temperature=0.6, top_p=0.95, top_k=20, do_sample=True,
)

# 加载lora权重
folder_path = "./output"
latest_chpoint = get_latest_chpoint(folder_path)


lora_model = PeftModel.from_pretrained(
    model=model, # 原始模型
    model_id= latest_chpoint # 微调得到的模型快照
)
# 模型合并
model_new = lora_model.merge_and_unload()

# 模型保存
model_new.save_pretrained("./Qwen/Qwen3-1.7B-LoRA")
tokenizer.save_pretrained("./Qwen/Qwen3-1.7B-LoRA")
print("模型保存成功!")