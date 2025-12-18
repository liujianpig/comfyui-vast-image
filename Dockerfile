FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git wget aria2 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --upgrade pip torch==2.1.0+cu121 torchvision==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html
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
EXPOSE 8188 8080 1111
CMD ["/bin/bash"]
