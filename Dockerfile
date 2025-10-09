FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip git && \
    apt-get clean

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install git+https://github.com/huggingface/diffusers.git
RUN pip3 install transformers==4.36.2 Pillow accelerate

# Clone the Qwen-Image-Edit repo
RUN git clone https://github.com/MozDevApps/Qwen-Image-Edit.git
WORKDIR /Qwen-Image-Edit

# Expose port if using a web interface (e.g., Gradio)
EXPOSE 7860

# Default command
CMD ["python3", "src/examples/demo.py"]
