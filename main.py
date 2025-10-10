from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
import torch

# Import the pipeline from HuggingFace's GitHub version of diffusers
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

app = FastAPI()

# Load the model
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16
).to("cuda")

@app.post("/generate")
async def generate_image(
    file1: UploadFile,
    file2: UploadFile,
    prompt: str = Form(...)
):
    image1 = Image.open(BytesIO(await file1.read())).convert("RGB")
    image2 = Image.open(BytesIO(await file2.read())).convert("RGB")

    inputs = {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": "",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        result = pipe(**inputs).images[0]

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
