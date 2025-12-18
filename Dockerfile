FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# 强制使用x86_64架构（避免arm64兼容问题）
ENV TORCH_INSTALL_URL=https://download.pytorch.org/whl/cu121
ENV PIP_DEFAULT_TIMEOUT=300
# 双源兜底（阿里云+清华）
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com,pypi.tuna.tsinghua.edu.cn

# 1. 修复基础环境（核心：补全编译/依赖库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential gcc g++ libjpeg-dev zlib1g-dev git wget aria2 ffmpeg systemd systemd-sysv \
    libssl-dev libffi-dev libgl1-mesa-glx libglib2.0-0 \
    # 补全CUDA依赖（关键：匹配cu121）
    libcudnn8 libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/* && \
    # 统一python命令，避免版本混乱
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3.10 /usr/bin/pip

# 2. 创建venv并升级基础工具（避免pip版本过低）
RUN python3.10 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel --no-cache-dir
ENV PATH=/venv/bin:$PATH

# 3. 核心修复：手动指定CUDA版torch的whl包（绕过pip源解析问题）
RUN pip install --no-cache-dir \
    # 直接指定cu121的whl包（适配Python3.10+Linux x86_64）
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    # 强制从PyTorch官方源下载（优先级高于PyPI）
    -f ${TORCH_INSTALL_URL}/torch_stable.html \
    # 禁用PyPI源，仅用指定的torch源（避免源冲突）
    --no-index

# 4. 验证torch+CUDA安装（提前失败，方便排查）
RUN python -c "import torch; \
    print(f'Torch版本: {torch.__version__}'); \
    print(f'CUDA可用: {torch.cuda.is_available()}'); \
    print(f'CUDA版本: {torch.version.cuda}'); \
    assert torch.cuda.is_available(), 'CUDA初始化失败！'"

# 5. 部署ComfyUI及依赖
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI && \
    cd /workspace/ComfyUI && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager && \
    pip install --no-cache-dir -r requirements.txt einops transformers accelerate safetensors

# 6. 安装ComfyUI自定义节点+补充依赖
RUN cd /workspace/ComfyUI/custom_nodes && \
    git clone https://github.com/THUDM/ComfyUI-CogVideoX-I2V && \
    git clone https://github.com/aigem/ComfyUI-Wav2Lip-FP16 && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/Fanno/ComfyUI-Video-Multicrop && \
    git clone https://github.com/Fanno/ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir face-alignment librosa opencv-python-headless

# 7. 部署GPT-SoVITS（独立venv，复用torch安装逻辑）
RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git /workspace/GPT-SoVITS && \
    python3.10 -m venv /workspace/GPT-SoVITS/venv && \
    /workspace/GPT-SoVITS/venv/bin/pip install --upgrade pip setuptools wheel --no-cache-dir && \
    # 复用torch官方源安装，避免重复踩坑
    /workspace/GPT-SoVITS/venv/bin/pip install --no-cache-dir \
        torch==2.1.0+cu121 torchaudio==2.1.0+cu121 \
        -f ${TORCH_INSTALL_URL}/torch_stable.html --no-index && \
    /workspace/GPT-SoVITS/venv/bin/pip install --no-cache-dir -r /workspace/GPT-SoVITS/requirements_infer.txt

# 8. 配置systemd服务
RUN echo '[Unit]\nDescription=ComfyUI\nAfter=network.target nvidia-persistenced.service\n[Service]\nType=simple\nUser=root\nGroup=root\nWorkingDirectory=/workspace/ComfyUI\nEnvironment="PATH=/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nEnvironment="PYTHONUNBUFFERED=1"\nExecStart=/venv/bin/python main.py --listen 0.0.0.0 --port 8188\nRestart=on-failure\nRestartSec=5s\nLimitNOFILE=65535\n[Install]\nWantedBy=multi-user.target' > /etc/systemd/system/comfyui.service && \
    systemctl enable comfyui

# 9. 下载小权重文件（增加重试机制）
RUN mkdir -p /workspace/ComfyUI/models/{ControlNet,loras,AnimateDiff,wav2lip,ipadapter} && \
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 --max-tries=3 -o /workspace/ComfyUI/models/loras/anime-film.safetensors \
      https://huggingface.co/liujianpig/anime-film/resolve/main/anime-film.safetensors && \
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 --max-tries=3 -o /workspace/ComfyUI/models/loras/flat-color.safetensors \
      https://huggingface.co/liujianpig/flat-color/resolve/main/flat-color.safetensors && \
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 --max-tries=3 -o /workspace/ComfyUI/models/AnimateDiff/mm-Stabilized-FPF-v2.ckpt \
      https://huggingface.co/guoyww/animatediff/resolve/main/mm-Stabilized-FPF-v2.ckpt && \
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 --max-tries=3 -o /workspace/ComfyUI/models/ControlNet/control_v11p_sd15_lineart.pth \
      https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth && \
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 --max-tries=3 -o /workspace/ComfyUI/models/ipadapter/ip-adapter_sd15.safetensors \
      https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors && \
    aria2c -x16 -c -s16 -k1M --timeout=600 --retry-wait=5 --max-tries=3 -o /workspace/ComfyUI/models/wav2lip/wav2lip_gfpkerfp16.pth \
      https://huggingface.co/aigem/wav2lip-fp16/resolve/main/wav2lip_gfpkerfp16.pth

# 10. 暴露端口+启动命令
EXPOSE 8188 8080 1111
CMD ["/bin/bash", "-c", "systemctl daemon-reload && systemctl start comfyui && tail -f /var/log/syslog"]
