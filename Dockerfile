FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential libjpeg-dev zlib1g-dev git wget aria2 ffmpeg systemd systemd-sysv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH
# 1. 确认用 python3.10 且先升级 pip
RUN python3.10 -m pip install --upgrade pip
# 2. 再装 torch/torchvision（分开写，避免并发编译）
RUN python3.10 -m pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.10 -m pip install torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
RUN cd /workspace/ComfyUI && git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager
RUN pip install -r /workspace/ComfyUI/requirements.txt einops transformers accelerate safetensors

RUN cd /workspace/ComfyUI/custom_nodes && \
    git clone https://github.com/THUDM/ComfyUI-CogVideoX-I2V && \
    git clone https://github.com/aigem/ComfyUI-Wav2Lip-FP16 && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/Fanno/ComfyUI-Video-Multicrop && \
    git clone https://github.com/Fanno/ComfyUI-Frame-Interpolation

RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git /workspace/GPT-SoVITS
RUN python3.10 -m venv /workspace/GPT-SoVITS/venv
RUN /workspace/GPT-SoVITS/venv/bin/pip install --upgrade pip torch==2.1.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN /workspace/GPT-SoVITS/venv/bin/pip install -r /workspace/GPT-SoVITS/requirements_infer.txt

RUN echo '[Unit]\nDescription=ComfyUI\nAfter=network.target\n[Service]\nType=simple\nUser=root\nWorkingDirectory=/workspace/ComfyUI\nEnvironment=PATH=/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\nExecStart=/venv/bin/python main.py --listen 0.0.0.0 --port 8188\nRestart=always\n[Install]\nWantedBy=multi-user.target' > /etc/systemd/system/comfyui.service
RUN systemctl enable comfyui
# ===== 大骨架同上（省略）=====

# 小权重（≤ 2 GB）直接装镜像
RUN mkdir -p /workspace/ComfyUI/models/{ControlNet,loras,AnimateDiff,wav2lip,ipadapter}

# ① 动漫常用 LoRA（≤ 200 MB × 5）
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/loras/anime-film.safetensors \
  https://huggingface.co/liujianpig/anime-film/resolve/main/anime-film.safetensors
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/loras/flat-color.safetensors \
  https://huggingface.co/liujianpig/flat-color/resolve/main/flat-color.safetensors

# ② AnimateDiff-SD1.5 运动模块（1.6 GB）
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/AnimateDiff/mm-Stabilized-FPF-v2.ckpt \
  https://huggingface.co/guoyww/animatediff/resolve/main/mm-Stabilized-FPF-v2.ckpt

# ③ ControlNet 动漫线稿（1.4 GB）
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/ControlNet/control_v11p_sd15_lineart.pth \
  https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth

# ④ IP-Adapter + Wav2Lip（< 1 GB）
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/ipadapter/ip-adapter_sd15.safetensors \
  https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors
RUN aria2c -x16 -c -o /workspace/ComfyUI/models/wav2lip/wav2lip_gfpkerfp16.pth \
  https://huggingface.co/aigem/wav2lip-fp16/resolve/main/wav2lip_gfpkerfp16.pth
EXPOSE 8188 8080 1111
CMD ["/bin/bash"]
