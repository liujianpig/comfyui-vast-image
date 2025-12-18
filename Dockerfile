FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

# 1. 修复基础依赖（补充缺失的库，解决编译/运行报错）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential libjpeg-dev zlib1g-dev git wget aria2 ffmpeg systemd systemd-sysv \
    libssl-dev libffi-dev libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# 2. 创建并激活全局venv（避免路径冲突）
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# 3. 安装PyTorch（指定cu121，国内源加速，避免编译失败）
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    --no-cache-dir

# 4. 部署ComfyUI及依赖（分步安装，避免缓存爆炸）
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI && \
    cd /workspace/ComfyUI && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager && \
    pip install --no-cache-dir -r requirements.txt einops transformers accelerate safetensors

# 5. 安装ComfyUI自定义节点（补充依赖，避免运行报错）
RUN cd /workspace/ComfyUI/custom_nodes && \
    git clone https://github.com/THUDM/ComfyUI-CogVideoX-I2V && \
    git clone https://github.com/aigem/ComfyUI-Wav2Lip-FP16 && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/Fanno/ComfyUI-Video-Multicrop && \
    git clone https://github.com/Fanno/ComfyUI-Frame-Interpolation && \
    # 补充自定义节点依赖
    pip install --no-cache-dir face-alignment librosa opencv-python-headless

# 6. 部署GPT-SoVITS（独立venv，避免依赖冲突）
RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git /workspace/GPT-SoVITS && \
    python3.10 -m venv /workspace/GPT-SoVITS/venv && \
    /workspace/GPT-SoVITS/venv/bin/pip install --upgrade pip setuptools wheel && \
    /workspace/GPT-SoVITS/venv/bin/pip install --no-cache-dir torch==2.1.0+cu121 torchaudio==2.1.0+cu121 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    /workspace/GPT-SoVITS/venv/bin/pip install --no-cache-dir -r /workspace/GPT-SoVITS/requirements_infer.txt

# 7. 配置systemd服务（修复权限/路径问题）
RUN echo '[Unit]\nDescription=ComfyUI\nAfter=network.target nvidia-persistenced.service\n[Service]\nType=simple\nUser=root\nGroup=root\nWorkingDirectory=/workspace/ComfyUI\nEnvironment="PATH=/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nEnvironment="PYTHONUNBUFFERED=1"\nExecStart=/venv/bin/python main.py --listen 0.0.0.0 --port 8188\nRestart=on-failure\nRestartSec=5s\nLimitNOFILE=65535\n[Install]\nWantedBy=multi-user.target' > /etc/systemd/system/comfyui.service && \
    systemctl enable comfyui

# 8. 下载小权重文件（添加超时重试，避免下载失败）
RUN mkdir -p /workspace/ComfyUI/models/{ControlNet,loras,AnimateDiff,wav2lip,ipadapter} && \
    # ① 动漫LoRA
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 -o /workspace/ComfyUI/models/loras/anime-film.safetensors \
      https://huggingface.co/liujianpig/anime-film/resolve/main/anime-film.safetensors && \
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 -o /workspace/ComfyUI/models/loras/flat-color.safetensors \
      https://huggingface.co/liujianpig/flat-color/resolve/main/flat-color.safetensors && \
    # ② AnimateDiff运动模块
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 -o /workspace/ComfyUI/models/AnimateDiff/mm-Stabilized-FPF-v2.ckpt \
      https://huggingface.co/guoyww/animatediff/resolve/main/mm-Stabilized-FPF-v2.ckpt && \
    # ③ ControlNet动漫线稿
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 -o /workspace/ComfyUI/models/ControlNet/control_v11p_sd15_lineart.pth \
      https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth && \
    # ④ IP-Adapter + Wav2Lip
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 -o /workspace/ComfyUI/models/ipadapter/ip-adapter_sd15.safetensors \
      https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors && \
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 -o /workspace/ComfyUI/models/wav2lip/wav2lip_gfpkerfp16.pth \
      https://huggingface.co/aigem/wav2lip-fp16/resolve/main/wav2lip_gfpkerfp16.pth

# 9. 暴露端口 + 启动命令（修复systemd运行问题）
EXPOSE 8188 8080 1111
CMD ["/bin/bash", "-c", "systemctl daemon-reload && systemctl start comfyui && tail -f /var/log/syslog"]
