#!/usr/bin/env python3
"""Script to download Qwen-Image-Edit-2509 models with progress tracking"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm


# Model configurations
MODELS = {
    "diffusion_models": [
        {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            "description": "Qwen Image Edit 2509 (FP8 Quantized)"
        }
    ],
    "text_encoders": [
        {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "description": "Qwen2.5-VL 7B Text Encoder (FP8)"
        }
    ],
    "vae": [
        {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "qwen_image_vae.safetensors",
            "description": "Qwen Image VAE"
        }
    ],
    "loras": [
        {
            "repo_id": "lightx2v/Qwen-Image-Lightning",
            "filename": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            "description": "Lightning LoRA (4-step)"
        },
        {
            "repo_id": "lightx2v/Qwen-Image-Lightning",
            "filename": "Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors",
            "description": "Lightning LoRA (8-step)"
        }
    ]
}


def download_model(repo_id: str, filename: str, local_dir: str, description: str):
    """Download a single model file"""
    try:
        print(f"\nDownloading: {description}")
        print(f"From: {repo_id}/{filename}")

        local_path = Path(local_dir) / filename

        # Check if already exists
        if local_path.exists():
            print(f"✓ Already exists: {local_path}")
            return True

        # Ensure directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

        print(f"✓ Downloaded to: {downloaded_path}")
        return True

    except Exception as e:
        print(f"✗ Error downloading {filename}: {str(e)}")
        return False


def main():
    """Main download function"""
    print("=" * 70)
    print("Qwen-Image-Edit-2509 Model Downloader")
    print("=" * 70)

    # Get ComfyUI path
    comfyui_path = os.environ.get('COMFYUI_PATH', '/workspace/ComfyUI')
    models_path = Path(comfyui_path) / 'models'

    print(f"\nModels directory: {models_path}")

    total_success = 0
    total_models = sum(len(models) for models in MODELS.values())

    # Download each category
    for category, models in MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"Category: {category}")
        print(f"{'=' * 70}")

        category_path = models_path / category

        for model in models:
            success = download_model(
                repo_id=model['repo_id'],
                filename=model['filename'],
                local_dir=str(category_path),
                description=model['description']
            )

            if success:
                total_success += 1

    # Summary
    print(f"\n{'=' * 70}")
    print("Download Summary")
    print(f"{'=' * 70}")
    print(f"Successfully downloaded: {total_success}/{total_models} models")

    if total_success == total_models:
        print("✓ All models downloaded successfully!")
        return 0
    else:
        print(f"✗ {total_models - total_success} models failed to download")
        return 1


if __name__ == "__main__":
    sys.exit(main())
