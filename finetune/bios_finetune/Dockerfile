# 使用 NVIDIA CUDA 基础镜像（包含编译工具链）
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget curl vim \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    ffmpeg libavcodec-extra \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# 配置 pip 使用清华源（加速下载，可选）
RUN pip3 config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装 PyTorch（版本与 CUDA 12.1 匹配）
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 Unsloth（使用官方预编译的wheel，避免Git元数据问题）
RUN pip3 install unsloth

# 安装 Jupyter Lab 和其他常用工具
RUN pip3 install jupyterlab ipywidgets

# 创建工作目录
WORKDIR /workspace

# 暴露 Jupyter Lab 端口
EXPOSE 8888

# 启动 Jupyter Lab（允许 root 运行，方便演示）
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]