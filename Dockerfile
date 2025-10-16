# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COMFYUI_PATH=/workspace/ComfyUI

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH}

# Install ComfyUI requirements
WORKDIR ${COMFYUI_PATH}
RUN apt-get update && apt-get install -y \
    python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    runpod \
    huggingface_hub \
    safetensors \
    accelerate \
    transformers \
    sentencepiece

# Create model directories
RUN mkdir -p ${COMFYUI_PATH}/models/diffusion_models \
    ${COMFYUI_PATH}/models/text_encoders \
    ${COMFYUI_PATH}/models/vae \
    ${COMFYUI_PATH}/models/loras \
    ${COMFYUI_PATH}/input \
    ${COMFYUI_PATH}/output

# Download Qwen-Image-Edit-2509 models
# Main diffusion model (FP8 quantized for lower VRAM)
RUN wget -O ${COMFYUI_PATH}/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors

# Text encoder (Qwen2.5-VL 7B FP8)
RUN wget -O ${COMFYUI_PATH}/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors

# VAE model
RUN wget -O ${COMFYUI_PATH}/models/vae/qwen_image_vae.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors

# Lightning LoRA for faster inference (4-steps)
RUN wget -O ${COMFYUI_PATH}/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors \
    https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors

# Optional: 8-step Lightning LoRA
RUN wget -O ${COMFYUI_PATH}/models/loras/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors \
    https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors

# Copy handler and workflow files
COPY handler.py ${COMFYUI_PATH}/handler.py
COPY qwen_edit_workflow.json ${COMFYUI_PATH}/qwen_edit_workflow.json
COPY download_models.py ${COMFYUI_PATH}/download_models.py

# Health check script
COPY healthcheck.py ${COMFYUI_PATH}/healthcheck.py

# Expose port
EXPOSE 8188

# Set working directory
WORKDIR ${COMFYUI_PATH}

# Run the handler
CMD ["python3", "handler.py"]
