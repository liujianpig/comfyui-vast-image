FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/venv/bin:$PATH

# 1. 安装基础依赖（最小集）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential gcc g++ wget git aria2 ffmpeg systemd systemd-sysv \
    libssl-dev libffi-dev libgl1-mesa-glx libglib2.0-0 libcudnn8 libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3.10 /usr/bin/pip

# 2. 创建venv并升级pip
RUN python3.10 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel --no-cache-dir

# 3. 核心：手动下载torch/cu121的whl包（Python3.10+x86_64）
RUN mkdir -p /tmp/torch_whl && cd /tmp/torch_whl && \
    # 下载torch核心包（约2.4GB）
    wget --no-check-certificate --timeout=600 --tries=3 \
      https://download.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl && \
    # 下载torchvision（约600MB）
    wget --no-check-certificate --timeout=600 --tries=3 \
      https://download.pytorch.org/whl/cu121/torchvision-0.16.0%2Bcu121-cp310-cp310-linux_x86_64.whl && \
    # 下载torchaudio（约200MB）
    wget --no-check-certificate --timeout=600 --tries=3 \
      https://download.pytorch.org/whl/cu121/torchaudio-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl && \
    # 离线安装whl包（彻底避开pip源问题）
    pip install --no-cache-dir *.whl && \
    # 清理临时文件
    rm -rf /tmp/torch_whl

# 4. 强制验证CUDA可用性（失败则终止构建）
RUN python -c "import torch; \
    assert torch.__version__ == '2.1.0+cu121', f'版本错误：{torch.__version__}'; \
    assert torch.cuda.is_available(), 'CUDA不可用！'; \
    assert torch.version.cuda == '12.1', f'CUDA版本错误：{torch.version.cuda}'; \
    print('✅ Torch+CUDA安装成功！')"

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

# 7. 部署GPT-SoVITS（复用离线whl逻辑）
RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git /workspace/GPT-SoVITS && \
    python3.10 -m venv /workspace/GPT-SoVITS/venv && \
    /workspace/GPT-SoVITS/venv/bin/pip install --upgrade pip setuptools wheel --no-cache-dir && \
    # 给GPT-SoVITS的venv也装torch（离线方式）
    mkdir -p /tmp/gpt_torch && cd /tmp/gpt_torch && \
    wget --no-check-certificate --timeout=600 --tries=3 \
      https://download.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl && \
    wget --no-check-certificate --timeout=600 --tries=3 \
      https://download.pytorch.org/whl/cu121/torchaudio-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl && \
    /workspace/GPT-SoVITS/venv/bin/pip install --no-cache-dir *.whl && \
    rm -rf /tmp/gpt_torch && \
    # 安装GPT-SoVITS依赖
    /workspace/GPT-SoVITS/venv/bin/pip install --no-cache-dir -r /workspace/GPT-SoVITS/requirements_infer.txt

# 8. 配置systemd服务
RUN echo '[Unit]\nDescription=ComfyUI\nAfter=network.target nvidia-persistenced.service\n[Service]\nType=simple\nUser=root\nGroup=root\nWorkingDirectory=/workspace/ComfyUI\nEnvironment="PATH=/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nEnvironment="PYTHONUNBUFFERED=1"\nExecStart=/venv/bin/python main.py --listen 0.0.0.0 --port 8188\nRestart=on-failure\nRestartSec=5s\nLimitNOFILE=65535\n[Install]\nWantedBy=multi-user.target' > /etc/systemd/system/comfyui.service && \
    systemctl enable comfyui

# 9. 下载小权重文件（增加重试）
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
