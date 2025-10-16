#!/usr/bin/env python3
"""Health check script for RunPod serverless"""

import requests
import sys
import time


def check_comfyui_health():
    """Check if ComfyUI is responding"""
    try:
        response = requests.get("http://127.0.0.1:8188/system_stats", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_models_exist():
    """Check if required models are present"""
    from pathlib import Path

    required_files = [
        "/workspace/ComfyUI/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        "/workspace/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "/workspace/ComfyUI/models/vae/qwen_image_vae.safetensors",
        "/workspace/ComfyUI/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors"
    ]

    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    return len(missing) == 0, missing


def main():
    """Run health checks"""
    print("Running health checks...")

    # Check models
    models_ok, missing = check_models_exist()
    if not models_ok:
        print("✗ Missing models:")
        for model in missing:
            print(f" - {model}")
        return 1
    else:
        print("✓ All required models present")

    # Check ComfyUI
    print("Checking ComfyUI server...")
    max_retries = 30
    for i in range(max_retries):
        if check_comfyui_health():
            print("✓ ComfyUI is responding")
            return 0

        if i < max_retries - 1:
            time.sleep(2)

    print("✗ ComfyUI is not responding")
    return 1


if __name__ == "__main__":
    sys.exit(main())
