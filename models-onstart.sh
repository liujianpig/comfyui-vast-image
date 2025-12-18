#!/bin/bash
source /workspace/venv/bin/activate
mkdir -p /workspace/ComfyUI/models/{CogVideoX,Wan2.2,LaVie} /workspace/ComfyUI/workflows

# CogVideoX 14 GB
aria2c -x16 -c -o /workspace/ComfyUI/models/CogVideoX/cogvideox-5b-i2v.fp16.safetensors \
  https://huggingface.co/THUDM/CogVideoX-5B-I2V/resolve/main/cogvideox-5b-i2v.fp16.safetensors

# Wan2.2-FLF2V fp8 7 GB
aria2c -x16 -c -o /workspace/ComfyUI/models/Wan2.2/wan2.2-flf2v-14B-fp8.safetensors \
  https://huggingface.co/ComfyUI/Wan2x-FLF2V/resolve/main/wan2.2-flf2v-14B-fp8.safetensors

# LaVie-FLF 4 GB
aria2c -x16 -c -o /workspace/ComfyUI/models/LaVie/lavie-flf-2B-fp16.safetensors \
  https://huggingface.co/LaVie-FLF/lavie-flf-2B-fp16.safetensors

systemctl restart comfyui
