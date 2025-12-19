# 匹配50系显卡：CUDA 12.4.1 + Ubuntu22.04（驱动由宿主机提供，无需容器内装）
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/venv/bin:$PATH
# 适配RTX 5070/5090算力（SM_90）
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# 优化显存调度
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. 修复apt源 + 安装基础依赖（移除容器内显卡驱动）
RUN apt-get update && \
    # 添加NVIDIA官方源（解决libcudnn9找不到的问题）
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    # 安装依赖（移除nvidia-driver/nvidia-settings，容器内无需装驱动）
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential gcc g++ git aria2 ffmpeg systemd systemd-sysv \
    libssl-dev libffi-dev libgl1-mesa-glx libglib2.0-0 \
    # 用CUDA源安装libcudnn9（适配12.4）
    libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/* && \
    # 统一Python/pip命令
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3.10 /usr/bin/pip

# 2. 创建venv并升级基础工具
RUN python3.10 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel --no-cache-dir

# 3. 离线安装PyTorch 2.4.0 + CUDA 12.4（50系显卡最优版本）
COPY ./torch_whl/*.whl /tmp/torch_whl/
RUN mkdir -p /tmp/torch_whl && \
    pip install --no-cache-dir /tmp/torch_whl/*.whl && \
    rm -rf /tmp/torch_whl

# 4. 验证50系显卡适配（强制校验算力）
RUN python -c "import torch; \
    assert torch.__version__ == '2.4.0+cu124', f'版本错误：{torch.__version__}'; \
    assert torch.cuda.is_available(), 'CUDA不可用！'; \
    assert torch.version.cuda == '12.4', f'CUDA版本错误：{torch.version.cuda}'; \
    # 验证50系显卡算力（SM_90）
    assert '9.0' in str(torch.cuda.get_device_capability(0)), f'算力不匹配：{torch.cuda.get_device_capability(0)}'; \
    print(f'✅ 显卡型号：{torch.cuda.get_device_name(0)}'); \
    print(f'✅ 算力版本：{torch.cuda.get_device_capability(0)}'); \
    print('✅ RTX 5070/5090 适配成功！')"

# 5. 部署ComfyUI及依赖（优化显存占用）
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI && \
    cd /workspace/ComfyUI && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager && \
    # 适配50系显卡，强制升级依赖
    pip install --no-cache-dir -r requirements.txt einops transformers accelerate safetensors --upgrade && \
    # 安装显存优化插件
    pip install --no-cache-dir xformers==0.0.27.post2 --upgrade

# 6. 安装ComfyUI自定义节点+50系适配
RUN cd /workspace/ComfyUI/custom_nodes && \
    git clone https://github.com/THUDM/ComfyUI-CogVideoX-I2V && \
    git clone https://github.com/aigem/ComfyUI-Wav2Lip-FP16 && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/Fanno/ComfyUI-Video-Multicrop && \
    git clone https://github.com/Fanno/ComfyUI-Frame-Interpolation && \
    # 补充节点依赖，适配50系显卡
    pip install --no-cache-dir face-alignment librosa opencv-python-headless --upgrade && \
    # 安装50系显卡专属优化库
    pip install --no-cache-dir nvidia-cudnn-cu12==9.1.0.70 --upgrade

# 7. 部署GPT-SoVITS（独立venv，适配50系）
RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git /workspace/GPT-SoVITS && \
    python3.10 -m venv /workspace/GPT-SoVITS/venv && \
    /workspace/GPT-SoVITS/venv/bin/pip install --upgrade pip setuptools wheel --no-cache-dir && \
    # 复用已下载的whl包
    mkdir -p /tmp/gpt_torch && \
    cp /tmp/torch_whl/torch-2.4.0+cu124-cp310-cp310-linux_x86_64.whl /tmp/gpt_torch/ && \
    cp /tmp/torch_whl/torchaudio-2.4.0+cu124-cp310-cp310-linux_x86_64.whl /tmp/gpt_torch/ && \
    /workspace/GPT-SoVITS/venv/bin/pip install --no-cache-dir /tmp/gpt_torch/*.whl && \
    rm -rf /tmp/gpt_torch && \
    # 安装GPT-SoVITS依赖，适配50系显卡
    /workspace/GPT-SoVITS/venv/bin/pip install --no-cache-dir -r /workspace/GPT-SoVITS/requirements_infer.txt --upgrade

# 8. 配置systemd服务（优化50系显卡显存）
RUN echo '[Unit]\nDescription=ComfyUI (RTX 5070/5090)\nAfter=network.target nvidia-persistenced.service\n[Service]\nType=simple\nUser=root\nGroup=root\nWorkingDirectory=/workspace/ComfyUI\nEnvironment="PATH=/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nEnvironment="PYTHONUNBUFFERED=1"\nEnvironment="TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0"\nEnvironment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"\nExecStart=/venv/bin/python main.py --listen 0.0.0.0 --port 8188 --cuda-malloc-backend cudaMallocAsync\nRestart=on-failure\nRestartSec=5s\nLimitNOFILE=65535\nLimitMEMLOCK=infinity\n[Install]\nWantedBy=multi-user.target' > /etc/systemd/system/comfyui.service && \
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
