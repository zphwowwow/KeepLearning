# BIOS 问答微调项目

基于 Qwen2.5-1.5B-Instruct 微调的 BIOS 领域问答模型。

## 环境要求

- Python 3.10
- RTX 3060 6GB（或更高）
- CUDA 11.8+

## 快速开始

```bash
# 1. 创建虚拟环境
conda create -n bios_finetune python=3.10 -y
conda activate bios_finetune

# 2. 安装依赖
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install unsloth transformers accelerate bitsandbytes
pip install "unsloth[cu118-torch271]"
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118


# 3. 准备数据（已提供 data/bios_data.json）
# 4. 开始微调
python scripts/train.py

# 5. 测试模型
python scripts/test.py