# ===== 基础镜像 =====
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ===== 系统依赖 =====
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential libjpeg-dev zlib1g-dev git wget aria2 ffmpeg systemd systemd-sysv \
    && rm -rf /var/lib/apt/lists/*

# ===== 16G swap =====
RUN fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile
RUN echo '/swapfile none swap sw 0 0' >> /etc/fstab

# ===== Python 虚拟环境 + PyTorch =====
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --upgrade pip
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ===== ComfyUI + Manager =====
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
RUN cd /workspace/ComfyUI && git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager
RUN pip install -r /workspace/ComfyUI/requirements.txt einops transformers accelerate safetensors

# ===== 节点：9宫格/对口型/IPA/多格/补帧 =====
RUN cd /workspace/ComfyUI/custom_nodes && \
    git clone https://github.com/THUDM/ComfyUI-CogVideoX-I2V && \
    git clone https://github.com/aigem/ComfyUI-Wav2Lip-FP16 && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/Fanno/ComfyUI-Video-Multicrop && \
    git clone https://github.com/Fanno/ComfyUI-Frame-Interpolation

# ===== GPT-SoVITS 推理环境（不装训练包）=====
RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git /workspace/GPT-SoVITS
RUN python3.10 -m venv /workspace/GPT-SoVITS/venv
RUN /workspace/GPT-SoVITS/venv/bin/pip install --upgrade pip torch==2.1.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN /workspace/GPT-SoVITS/venv/bin/pip install -r /workspace/GPT-SoVITS/requirements_infer.txt

# ===== 预装大模型 & 工作流 =====
RUN mkdir -p /workspace/ComfyUI/models/{CogVideoX,wav2lip,ipadapter} /workspace/ComfyUI/workflows /workspace/GPT-SoVITS/models

# ① 动漫 9 宫格视频模型 + 工作流
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/CogVideoX/cogvideox-5b-i2v.fp16.safetensors \
    https://huggingface.co/THUDM/CogVideoX-5B-I2V/resolve/main/cogvideox-5b-i2v.fp16.safetensors
RUN wget -P /workspace/ComfyUI/workflows \
    https://huggingface.co/THUDM/CogVideoX-5B-I2V/resolve/main/workflows/cogx_anime_9grid_16s.json

# ② 锁脸模型
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/ipadapter/ip-adapter_sd15.safetensors \
    https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors

# ③ 对口型模型
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/wav2lip/wav2lip_gfpkerfp16.pth \
    https://huggingface.co/aigem/wav2lip-fp16/resolve/main/wav2lip_gfpkerfp16.pth

# ④ 中文女声音色（GPT-SoVITS 推理用）
RUN aria2c -x16 -c -o /workspace/GPT-SoVITS/models/gsv-cn-female.pth \
    https://huggingface.co/spaces/OrdinaryMemory/GPT-SoVITS/resolve/main/gsv-cn-female.pth

# ⑤ 一键工作流（9宫格+对口型+TTS 已连好）
RUN wget -P /workspace/ComfyUI/workflows \
    https://huggingface.co/aigem/wav2lip-fp16/resolve/main/workflows/anime_9grid_lip_tts.json

# ===== systemd 自启 ComfyUI =====
RUN echo '[Unit]\nDescription=ComfyUI\nAfter=network.target\n[Service]\nType=simple\nUser=root\nWorkingDirectory=/workspace/ComfyUI\nEnvironment=PATH=/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\nExecStart=/venv/bin/python main.py --listen 0.0.0.0 --port 8188\nRestart=always\n[Install]\nWantedBy=multi-user.target' > /etc/systemd/system/comfyui.service
RUN systemctl enable comfyui

EXPOSE 8188 8080 1111
CMD ["/bin/bash"]
