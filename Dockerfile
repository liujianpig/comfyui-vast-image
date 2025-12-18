# 使用官方存在的 CUDA 12.1.1 标签
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. 系统依赖（含 Python 3.10 开发包）
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential libjpeg-dev zlib1g-dev git wget aria2 ffmpeg \
    systemd systemd-sysv \
    && rm -rf /var/lib/apt/lists/*

# 2. 16G swap（小内存实例也扛得住）
RUN fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile
RUN echo '/swapfile none swap sw 0 0' >> /etc/fstab

# 3. Python 虚拟环境 + PyTorch（正确版本组合）
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --upgrade pip
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 4. ComfyUI + Manager
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
RUN cd /workspace/ComfyUI && git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager
RUN pip install -r /workspace/ComfyUI/requirements.txt einops transformers accelerate safetensors

# 5. 视频/对口型/多格/IPA 节点（只装代码，不下载权重）
RUN cd /workspace/ComfyUI/custom_nodes && \
    git clone https://github.com/THUDM/ComfyUI-CogVideoX-I2V && \
    git clone https://github.com/aigem/ComfyUI-Wav2Lip-FP16 && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/Fanno/ComfyUI-Video-Multicrop && \
    git clone https://github.com/Fanno/ComfyUI-Frame-Interpolation

# 6. GPT-SoVITS 推理环境（不装训练包，省 5 GB）
RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git /workspace/GPT-SoVITS
RUN python3.10 -m venv /workspace/GPT-SoVITS/venv
RUN /workspace/GPT-SoVITS/venv/bin/pip install --upgrade pip torch==2.1.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN /workspace/GPT-SoVITS/venv/bin/pip install -r /workspace/GPT-SoVITS/requirements_infer.txt

# 7. systemd 自启 ComfyUI
COPY comfyui.service /etc/systemd/system/
RUN systemctl enable comfyui

EXPOSE 8188 8080 1111
CMD ["/bin/bash"]
