"""
多模态LoRA微调-Qwen3-VL-4B-Instruct
"""
import transformers
from modelscope import snapshot_download
from transformers import AutoProcessor,Trainer
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import torch
import json
import random
import sys
import os
import re
from pathlib import Path
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Sequence,Dict

# 添加依赖：将qwenvl加入PYTHONPATH，使得tranier、qwenvl下的内容可被导入
sys.path.append("/root/autodl-tmp/qwenvl")
from data.rope2d import get_rope_index_3

data_path = "/root/autodl-tmp/data"  # 图片数据路径
annotation_path = "/root/autodl-tmp/data/train_data.jsonl" # 训练集路径
eval_path = "/root/autodl-tmp/data/val_data.jsonl"  #验证集路径

from train.argument import TrainingArguments,ModelArguments,DataArguments

# 数据相关参数
data_args=DataArguments(
    #缩放图片尺寸，保证处理器接收的像素总数落在一个区间内，避免高分辨率图片导致内存爆炸
    max_pixels = 50176,
    min_pixels = 784,
)

# 模型相关参数
model_args = ModelArguments(
    model_name_or_path="Qwen/Qwen3-VL-4B-Instruct",  #模型名称
    tune_mm_llm=True,  #控制语言模型组件是否参与训练
    tune_mm_mlp=True,  #控制多模态投影器是否参与训练（Merge模块）
    tune_mm_vision=False  #控制视觉编码器是否参与训练
)

#训练相关参数
training_args = TrainingArguments(
    cache_dir = "/root/autodl-tmp/",  #下载模型的保存路径
    output_dir="/root/autodl-tmp/output",  #训练结果（包括checkpoint和微调后的模型）输出路径
    dataloader_num_workers=4,  #并行加载数据的子进程数，可加速数据预处理与加载过程
    bf16=True,  #启用 bfloat16 精度训练，用flashAttention2必须开
    
    lora_r=64,  #低秩维度
    lora_alpha=128,  #缩放因子
    lora_dropout=0.0,  #正则化

    num_train_epochs = 2,  #训练轮次
    learning_rate = 1e-5,  #学习率
    lr_scheduler_type="cosine",  #控制训练过程中学习率的调度策略
    weight_decay=0,  #模型权重衰减，用于防止模型过拟合
    warmup_ratio=0.03,  #控制学习率预热的比例，即训练初期逐步增加学习率的步数占总训练步数的比例
    max_grad_norm=1,  #用于梯度裁剪，防止训练过程中梯度过大导致的训练不稳定
    optim = "adamw_torch",  #优化器类型，可替换为adamw_bnb_8bit、fused optimizers
    model_max_length = 2048,  #序列最大长度，控制tokenizer阶段、RoPE范围以及显存占用，序列会被填充到该长度或从该长度截断
    per_device_train_batch_size=4,  #各GPU上的批大小，即每次前向传播处理4个样本
    gradient_accumulation_steps=4,  #梯度累积步数，即每积累4次梯度更新1次参数
    # save_only_model=True,  #仅保存训练后的模型，不保存Adapter状态、训练状态等
    # gradient_checkpoint=True,  #启用梯度检查点，以时间换空间
    eval_strategy="no",  #表示禁用训练过程中的评估，可有效加快训练过程
    # eval_steps=500,  #每500步进行一次评估，配合eval_strategy="steps"使用
    save_strategy="steps",  #按步数间隔保存checkpoint
    save_steps=1000,  #每隔1000步保存一次checkpoint
    save_total_limit=1,  #checkpoint保留个数
    logging_steps=1,  #每一步都在日志输出训练指标，方便实时监控情况
    run_name="qwen3-vl-4b-lora",  
    logging_dir="/root/tf-logs/",  #指定日志输出目录
    report_to="tensorboard",  #指定训练日志和指标的上报平台
)

os.makedirs(training_args.output_dir, exist_ok=True)  #创建输出文件夹

# 从modelscope上下载模型到本地目录
model_dir = snapshot_download(model_args.model_name_or_path,
                              local_dir=training_args.cache_dir+model_args.model_name_or_path,  #模型本地缓存路径
                             revision="master")

# 加载模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, 
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  #一种加速注意力计算并减少内存占用的算法（可能硬件不支持），兼容性最好eager、性能兼容性均衡sdpa
    dtype=torch.float16,
    do_sample=True,top_p=0.8,top_k=20,temperature=0.7  #官方推荐使用的参数
)
model.config.use_cache = False  #训练时不需要自回归生成的KV缓存，减少内存占用
model.enable_input_require_grads()  #确保LoRA训练过程中可顺利反向传播梯度

# 加载处理器
processor = AutoProcessor.from_pretrained(
    model_dir,
    trust_remote_code=True,
)

processor.image_processor.min_pixels = data_args.min_pixels,
processor.image_processor.max_pixels = data_args.max_pixels,
# processor.image_processor.merge_size = 2  #压缩视觉token数

#加载分词器
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_dir,
    model_max_length=2048,
    padding_side="right",  #在原token的右侧填充，确保序列长度一致，且不影响注意力机制的计算
    use_fast=False, #Fast tokenizer在多模态场景下容易出bug，建议不启用
)

# 构造消息模板（单条记录 → 消息模板）
def build_messages(sample,base_path):
    images = [sample.get("image")]  # 提取样本中的image
    image_pool = [] # 构建多媒体池存放提取到的所有image
    for img in images:
        image_pool.append({"type":"image","image":str(Path(base_path).joinpath(img))})  #转换为绝对路径
    
    messages = []
    for turn in sample["conversations"]:  # 适配多轮对话
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]
        
        #判断对话中的角色，只有user消息才包含特殊占位符，assistant消息只是纯文本
        if role=="user":
            content = []
            # 通过<image>占位符拆分文本，同时保留分隔符
            text_parts = re.split(r"(<image>)", text)
            
            for seg in text_parts: #判断标签数是否与图片数一致
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # 模型回复消息只包含文本
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
        
        # 检查未使用的媒体文件
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    
    return messages
    

from qwen_vl_utils import process_vision_info

IGNORE_INDEX = -100 # CrossEntropyLoss忽略的token
IMAGE_TOKEN_INDEX = 151655  # 对应tokenizer中的<image>特殊token
DEFAULT_IMAGE_TOKEN = "<image>"  # 数据中使用的图片占位符

# 消息模板 → token + label 
def preprocess_message(sample,processor,data_path):
    base_path = Path(data_path)  
    messages = build_messages(sample,base_path)  #对每个样本按消息模板进行处理
    
    # 与Qwen3的apply_chat_template同理，将message处理为token（图片由<image>占位符代替）
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 调用工具库中的process_vision_info，快速处理视觉输入数据
    images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    video_metadatas = None #没有视频输入，所以设为空
    
    #调用processor，得到input_ids、attention_mask、pixel_values、image_grid_thw 
    result = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
    
    input_ids = torch.tensor(result['input_ids'])
    #将labels按inputs_ids的长度全设为-100，表示默认所有token不参与loss计算
    labels = torch.full_like(input_ids,IGNORE_INDEX) 
    input_ids_flat = input_ids[0].tolist()
    L=len(input_ids_flat)  #计算token长度
    pos = 0  #游标
    while pos < L:
        # 以特殊token为边界，只将assistant的回答拷贝到labels，将user/system/image token全部mask掉
        if input_ids_flat[pos] == 77091:  #77091=assistant
            ans_start = pos + 2
            ans_end = ans_start
            # 从assistant标记后的内容起算，碰到<im_end>为止,计算停止位置
            while ans_end < L and input_ids_flat[ans_end] != 151645:  #151645=<|im_end|>
                ans_end += 1
            if ans_end < L:
                #将labels中回复内容对应位置的值从-100还原回真正的token id
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1
    result["labels"] = labels
    result["input_ids"] = input_ids
    # 此时inputs_ids就是所有输入内容，而labels就是完成了遮蔽的张量
    return result
    
    
# 定义数据类，完成数据读取与处理
class ImgDataSet(Dataset):
    # 读取图文数据集
    def __init__(self,processor,data_path,annotation_path):
        self.processor=processor
        self.tokenizer=processor.tokenizer
        self.data_path = data_path
        self.samples=[]
        with open(annotation_path,"r",encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        
    # 计算数据量
    def __len__(self):
        return len(self.samples)
    
    # 单个样本的处理，将图片+问答对 → token+label
    def __getitem__(self,idx):
        sample = self.samples[idx]
        # 调用函数处理样本，得到token + label
        data_dict = preprocess_message(sample,self.processor,self.data_path)
        seq_len = data_dict["input_ids"][0].size(0)
        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None
        
        # 调用get_rope_index生成position_ids
        position_ids, _ = get_rope_index_3(
            getattr(processor.image_processor, "merge_size", 2),
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        )
        # 将位置编码加入到data_dict中
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]  #构造简单的attention_mask，只包含序列长度信息
        
        # 解码输入序列为文本(以便进行调试和检查) 
        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False  # 保留特殊token以便理解完整序列
        )
        
        # 将-100(IGNORE_INDEX)替换为pad_token_id，因为tokenizer无法解码-100
        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        # 解码labels（以便调试）
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)
        
        return data_dict
        
        
# 完成position_ids的填充与拼接，
# tensor_list是包含多个position_ids张量的列表，形状遵循[batch_size, 3, seq_len]
def pad_and_cat(tensor_list):
    # 确认输入张量中序列长度的最大值
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]  #计算填充长度
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)  #将填充部分全部设为1
        padded_tensors.append(padded_tensor)
    
    # 在batch维度（第1维）上拼接所有张量，拼接后的张量形状[total_batch_size, 3, max_seq_len]  
    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """微调数据整理器，将多个训练实例整理为一个batch，使得一次前向传播可同时处理多个样本"""
    
    tokenizer: transformers.PreTrainedTokenizer #参数：整理器使用的分词器

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 从实例中提取input_ids、labels、position_ids
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        # 移除batch维度，为padding做准备，因为每个instance来自preprocess_message，形状为[1, seq_len]
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        # padding，统一长度
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id  #用pad_token_id做填充
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX  #用-100做填充
        )
        # 调用pad_and_cat对position_ids进行处理
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        # 构建batch字典
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        # 处理图像数据
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        # 将图像数据添加到batch中
        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["position_ids"] = position_ids
        return batch
        

train_dataset = ImgDataSet(processor,data_path,annotation_path)
eval_dataset = ImgDataSet(processor,data_path,eval_path)
data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
data_module = dict(
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator=data_collator
)


#冻结所有参数
for p in model.parameters():
    p.requires_grad = False

# LoRA配置
lora_config = LoraConfig(
    r = training_args.lora_r,
    lora_alpha=training_args.lora_alpha,
    lora_dropout=training_args.lora_dropout,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],  #选定要针对哪些线性层进行微调
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model,lora_config)  #将LoRA架构融合到基座模型中

# 构建trainer，设定具体要微调的模型、使用的数据集等
trainer = Trainer(
	model=model,
    processing_class=tokenizer,
    args=training_args,
    **data_module
)

# 开始训练
trainer.train()

trainer.save_state()  #保存训练器状态（包括训练参数、优化器状态等元信息）
model.config.use_cache=True  #恢复模型配置中的缓存设置，训练完后启用KV缓存来支持推理
torch.cuda.synchronize()  #同步所有CUDA操作，确保GPU上的计算都已完成
trainer.save_model(training_args.output_dir)  #保存模型到输出目录

## 若因意外原因导致无法通过以上步骤保存模型训练器状态，亦可通过checkpoint合并模型
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

#基座模型
model_path = "/root/autodl-tmp/Qwen/Qwen3-VL-4B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="auto", device_map="auto",do_sample=True,top_p=0.8,top_k=20,temperature=0.7
)
processor = AutoProcessor.from_pretrained(model_path)

# 加载lora权重
latest_chpoint ="/root/autodl-tmp/output/checkpoint-8282"  #名称按照最新的checkpoint目录修改

lora_model = PeftModel.from_pretrained(
    model=model, # 原始模型
    model_id= latest_chpoint # 微调得到的模型快照
)

model_new = lora_model.merge_and_unload()

# 经过模型效果验证后，即可进行模型保存
model_new.save_pretrained("/root/autodl-tmp/Qwen/Qwen3-VL-4B-LoRA")
processor.save_pretrained("/root/autodl-tmp/Qwen/Qwen3-VL-4B-LoRA")
print("模型保存成功~")