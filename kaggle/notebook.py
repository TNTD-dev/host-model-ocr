# =============================================================================
# Kaggle Notebook: Qwen2.5-VL-7B-Instruct OCR Server
# =============================================================================
# Copy this entire file into a Kaggle notebook and run all cells.
# Make sure to enable GPU in notebook settings (T4 x2).
# =============================================================================

# =============================================================================
# Cell 1: Install Dependencies
# =============================================================================
# !pip install fastapi uvicorn python-multipart pyngrok
# !pip install git+https://github.com/huggingface/transformers accelerate qwen-vl-utils==0.0.8

# =============================================================================
# Cell 2: Import Libraries
# =============================================================================
import io
import base64
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pyngrok import ngrok, conf
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =============================================================================
# Cell 3: Configuration
# =============================================================================

NGROK_AUTHTOKEN = "YOUR_NGROK_TOKEN_HERE"  # <-- Replace with your token
PORT = 8000
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROMPT = "Extract all text from this image"

# =============================================================================
# Cell 4: Load Model and Processor
# =============================================================================

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # T4 GPU: use float16, NOT bfloat16
    device_map="auto",
)

min_pixels = 256 * 28 * 28     # ~200k pixels
max_pixels = 1280 * 28 * 28    # ~1M pixels (256-1280 visual tokens)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

print("Model loaded successfully!")

# =============================================================================
# Cell 5: Define FastAPI App
# =============================================================================

app = FastAPI(title="Qwen2.5-VL OCR Server")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    prompt: str = Form(default=DEFAULT_PROMPT),
):
    start_time = time.time()

    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image bytes
    image_bytes = await image.read()

    # Convert to PIL Image for validation
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))  # Re-open after verify
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Convert to base64 for model input
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    # Build messages for Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_uri},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Run inference
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    processing_time = time.time() - start_time

    return JSONResponse(
        content={
            "text": output_text.strip(),
            "processing_time": round(processing_time, 2),
        }
    )


# =============================================================================
# Cell 6: Start Server with ngrok Tunnel
# =============================================================================

# Start uvicorn in a daemon thread
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


thread = threading.Thread(target=run_server, daemon=True)
thread.start()

# Wait for server to start
time.sleep(2)

# Open ngrok tunnel
conf.get_default().auth_token = NGROK_AUTHTOKEN
public_url = ngrok.connect(PORT).public_url

print(f"=" * 60)
print(f"Server is running!")
print(f"Public ngrok URL: {public_url}")
print(f"Health check:     {public_url}/health")
print(f"=" * 60)
print(f"Copy this URL and paste it into your .env file as KAGGLE_NGROK_URL")
