import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    KAGGLE_NGROK_URL,
    KAGGLE_TIMEOUT,
    MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
)
from app.schemas import OCRResponse, HealthResponse

app = FastAPI(title="OCR Proxy Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_image(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    image: UploadFile = File(...),
    prompt: str = Form(default="Extract all text from this image"),
):
    if not KAGGLE_NGROK_URL:
        raise HTTPException(
            status_code=503,
            detail="KAGGLE_NGROK_URL not configured. Set it in .env file.",
        )

    _validate_image(image)

    image_bytes = await image.read()

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB",
        )

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=KAGGLE_TIMEOUT) as client:
            files = {"image": (image.filename, image_bytes, image.content_type)}
            data = {"prompt": prompt}

            response = await client.post(
                f"{KAGGLE_NGROK_URL}/predict",
                files=files,
                data=data,
            )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Kaggle request timed out after {KAGGLE_TIMEOUT}s. Model may be loading.",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Kaggle server. Check if notebook is running.",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error calling Kaggle: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Kaggle returned error: {response.status_code} - {response.text}",
        )

    result = response.json()
    total_time = time.time() - start_time

    return OCRResponse(
        text=result.get("text", ""),
        processing_time=round(total_time, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    if not KAGGLE_NGROK_URL:
        return HealthResponse(status="ok", kaggle_reachable=False)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{KAGGLE_NGROK_URL}/health")
            kaggle_ok = resp.status_code == 200
    except Exception:
        kaggle_ok = False

    return HealthResponse(status="ok", kaggle_reachable=kaggle_ok)
