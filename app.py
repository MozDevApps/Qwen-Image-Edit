from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from qwen_image.model import QwenImageModel

app = FastAPI()

# Load the Qwen model once on startup
model = QwenImageModel.from_pretrained("Qwen-Image-Edit-2509")
model.eval()

@app.post("/edit")
async def edit_image(file: UploadFile = File(...)):
    # Read uploaded image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Run the model (adapt the method according to repo API)
    edited_img = model.edit(img)  

    # Return edited image as bytes
    buf = io.BytesIO()
    edited_img.save(buf, format="PNG")
    buf.seek(0)
    return {"image_bytes": buf.getvalue()}
