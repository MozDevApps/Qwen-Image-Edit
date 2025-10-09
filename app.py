from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
from diffusers import QwenImageEditPlusPipeline

app = FastAPI(title="Qwen-Image Outfit Transfer API")

# Load pipeline once on startup
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=True)

@app.post("/edit")
async def edit_outfit(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    prompt: str = Form("The woman in image 1 now wears the outfit from image 2")
):
    """
    Accepts two images (image1: person, image2: outfit)
    and runs prompt-based editing.
    Returns the edited image as PNG.
    """
    # Read uploaded images
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()
    img1 = Image.open(io.BytesIO(img1_bytes)).convert("RGB")
    img2 = Image.open(io.BytesIO(img2_bytes)).convert("RGB")

    # Prepare inputs for QwenImageEditPlusPipeline
    inputs = {
        "image": [img1, img2],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }

    # Run inference
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]

    # Convert to bytes
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
