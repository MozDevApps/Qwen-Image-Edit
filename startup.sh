#!/bin/bash
# startup.sh - Verify models and start handler
set -e

echo "=========================================="
echo "ComfyUI Qwen-Image-Edit Startup"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found. Wrong directory?"
    exit 1
fi

echo "✓ Working directory: $(pwd)"

# Verify all required models exist
echo ""
echo "Checking models..."

MODELS_OK=true

check_model() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo "  ✓ $2: $SIZE"
    else
        echo "  ✗ MISSING: $2"
        echo "    Expected: $1"
        MODELS_OK=false
    fi
}

check_model "models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" "Diffusion Model"
check_model "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" "Text Encoder"
check_model "models/vae/qwen_image_vae.safetensors" "VAE"
check_model "models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors" "LoRA 4-step"
check_model "models/loras/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors" "LoRA 8-step"

if [ "$MODELS_OK" = false ]; then
    echo ""
    echo "ERROR: Some models are missing!"
    echo "Models should be downloaded during Docker build."
    echo "Please rebuild the Docker image."
    exit 1
fi

echo ""
echo "✓ All models present"

# Check disk space
echo ""
echo "Disk space:"
df -h / | tail -n 1

# Check GPU
echo ""
echo "GPU check:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not found"
fi

# Check Python and packages
echo ""
echo "Python version: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Create necessary directories
mkdir -p input output temp

echo ""
echo "=========================================="
echo "Starting handler..."
echo "=========================================="
echo ""

# Start the handler with unbuffered output
exec python3 -u handler.py
