# Feature: OCR Tool with Qwen2.5-VL-7B-Instruct

## Feature Description

Build a local OCR tool for developers that extracts text from images using the Qwen2.5-VL-7B-Instruct vision-language model hosted on Kaggle GPU. The system has three components: a Kaggle notebook running the model behind a FastAPI server with ngrok tunnel, a local FastAPI backend that proxies requests to Kaggle, and a Streamlit UI for uploading images and viewing OCR results.

## User Story

As a developer, I want to upload an image and extract text from it using a powerful vision-language model, so that I can quickly OCR documents, handwriting, or screenshots without setting up GPU infrastructure locally.

## Problem Statement

Running large vision-language models locally requires expensive GPU hardware. Kaggle provides free GPU access (T4 x2) but notebooks don't expose public APIs. Developers need a simple UI to upload images and get OCR results without managing infrastructure.

## Solution Statement

- **Kaggle notebook**: Run Qwen2.5-VL-7B-Instruct on T4 GPU, expose via FastAPI + ngrok tunnel
- **Local FastAPI**: Proxy requests to Kaggle, handle validation, provide health check
- **Streamlit UI**: Simple drag-and-drop interface for image upload and OCR result display

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: Kaggle notebook, local FastAPI server, Streamlit UI
**Dependencies**: transformers, qwen-vl-utils, fastapi, uvicorn, pyngrok, streamlit, httpx, python-multipart, Pillow, torch

---

## CONTEXT REFERENCES

### Relevant Codebase Files

This is a greenfield project — no existing files. All files are new.

### New Files to Create

- `kaggle/notebook.py` — Kaggle notebook: FastAPI server + model inference + ngrok tunnel
- `app/main.py` — Local FastAPI server: proxy to Kaggle, health check
- `app/config.py` — Settings: ngrok URL, timeout, file size limit
- `app/schemas.py` — Pydantic response models
- `streamlit_app/app.py` — Streamlit UI: upload, preview, OCR, copy
- `requirements.txt` — Local dependencies
- `requirements-kaggle.txt` — Kaggle notebook dependencies
- `.env.example` — Environment variable template
- `README.md` — Setup and usage instructions

### Relevant Documentation

- [Qwen2.5-VL-7B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
  - Model card with usage examples
  - Why: Primary model for OCR inference
- [qwen-vl-utils](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-utils)
  - Image preprocessing utilities
  - Why: Required for `process_vision_info()` to handle image inputs
- [FastAPI File Upload Docs](https://fastapi.tiangolo.com/tutorial/request-files/)
  - Multipart file upload pattern
  - Why: Both local and Kaggle FastAPI need file upload endpoints
- [pyngrok](https://pyngrok.readthedocs.io/en/latest/)
  - Python wrapper for ngrok
  - Why: Tunnel from Kaggle notebook to public internet
- [Streamlit File Uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
  - File upload widget
  - Why: Image upload UI component

### Patterns to Follow

**FastAPI endpoint pattern:**
```python
@app.post("/ocr")
async def ocr(image: UploadFile, prompt: str = Form(default="Extract all text from this image")):
    # validate file type
    # validate file size
    # call Kaggle endpoint
    # return result
```

**Model inference pattern (Qwen2.5-VL):**
```python
messages = [{"role": "user", "content": [
    {"type": "image", "image": image_bytes_or_path},
    {"type": "text", "text": prompt}
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = inputs.to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

**ngrok tunnel pattern:**
```python
from pyngrok import ngrok, conf
conf.get_default().auth_token = NGROK_AUTHTOKEN
public_url = ngrok.connect(8000).public_url
```

**Gotchas:**
- T4 GPU: use `torch.float16`, NOT `torch.bfloat16` (T4 has no native BF16)
- Install transformers from source: `pip install git+https://github.com/huggingface/transformers`
- Set `min_pixels`/`max_pixels` on processor to avoid OOM on T4 (256-1280 visual tokens)
- Trim generated_ids to remove input portion before decoding
- Use `daemon=True` thread for uvicorn in Kaggle notebook
- Start uvicorn BEFORE opening ngrok tunnel

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation

**Tasks:**
- Create project structure (directories, __init__.py files)
- Create requirements files for local and Kaggle
- Create .env.example with configuration variables
- Create config.py with settings management

### Phase 2: Kaggle Notebook

**Tasks:**
- Create kaggle/notebook.py with full model loading, FastAPI server, and ngrok setup
- Include health check endpoint
- Include /predict endpoint with image + prompt input
- Handle model loading, inference, and error cases

### Phase 3: Local FastAPI Backend

**Tasks:**
- Create app/main.py with FastAPI app
- Create POST /ocr endpoint that proxies to Kaggle
- Create GET /health endpoint that checks Kaggle connectivity
- Add file validation (type, size) and error handling
- Add timeout handling for Kaggle requests

### Phase 4: Streamlit UI

**Tasks:**
- Create streamlit_app/app.py
- Image upload with preview
- Optional prompt input with default
- Extract button with loading spinner
- Result display with copy functionality
- Processing time display

### Phase 5: Documentation

**Tasks:**
- Create README.md with setup instructions
- Document Kaggle notebook setup steps
- Document ngrok configuration
- Document how to run all components

---

## STEP-BY-STEP TASKS

### Task 1: CREATE project structure

- **CREATE** directories: `app/`, `streamlit_app/`, `kaggle/`
- **CREATE** empty `__init__.py` in `app/`
- **VALIDATE**: `ls -la app/ streamlit_app/ kaggle/`

### Task 2: CREATE requirements-local.txt

- **IMPLEMENT**: Local dependencies list
- **CONTENT**:
  ```
  fastapi==0.115.6
  uvicorn==0.34.0
  python-multipart==0.0.18
  httpx==0.28.1
  streamlit==1.41.1
  python-dotenv==1.0.1
  Pillow==11.1.0
  ```
- **VALIDATE**: `cat requirements-local.txt`

### Task 3: CREATE requirements-kaggle.txt

- **IMPLEMENT**: Kaggle notebook dependencies
- **CONTENT**:
  ```
  fastapi==0.115.6
  uvicorn==0.34.0
  python-multipart==0.0.18
  pyngrok==7.2.3
  git+https://github.com/huggingface/transformers
  accelerate
  qwen-vl-utils==0.0.8
  torch
  Pillow==11.1.0
  ```
- **VALIDATE**: `cat requirements-kaggle.txt`

### Task 4: CREATE .env.example

- **IMPLEMENT**: Environment variable template
- **CONTENT**:
  ```
  KAGGLE_NGROK_URL=https://your-ngrok-url.ngrok-free.app
  KAGGLE_TIMEOUT=120
  MAX_FILE_SIZE_MB=10
  ```
- **VALIDATE**: `cat .env.example`

### Task 5: CREATE app/config.py

- **IMPLEMENT**: Settings class using pydantic-settings or os.getenv
- **PATTERN**: Simple config with defaults
- **IMPORTS**: `os`, `dotenv`
- **CONTENT**:
  ```python
  import os
  from dotenv import load_dotenv

  load_dotenv()

  KAGGLE_NGROK_URL = os.getenv("KAGGLE_NGROK_URL", "")
  KAGGLE_TIMEOUT = int(os.getenv("KAGGLE_TIMEOUT", "120"))
  MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
  MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
  ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
  ```
- **VALIDATE**: `python -c "from app.config import KAGGLE_NGROK_URL; print(KAGGLE_NGROK_URL)"`

### Task 6: CREATE app/schemas.py

- **IMPLEMENT**: Pydantic response models
- **IMPORTS**: `pydantic`
- **CONTENT**:
  ```python
  from pydantic import BaseModel

  class OCRResponse(BaseModel):
      text: str
      processing_time: float

  class HealthResponse(BaseModel):
      status: str
      kaggle_reachable: bool
  ```
- **VALIDATE**: `python -c "from app.schemas import OCRResponse; print(OCRResponse)"`

### Task 7: CREATE kaggle/notebook.py

- **IMPLEMENT**: Complete Kaggle notebook code
- **PATTERN**: FastAPI + ngrok + model inference (see Patterns section above)
- **GOTCHA**: Use `torch.float16` for T4, set `min_pixels`/`max_pixels`, install transformers from source
- **STRUCTURE**:
  1. Install dependencies cell
  2. Load model and processor cell
  3. Define FastAPI app with `/predict` and `/health` endpoints
  4. Start uvicorn in daemon thread
  5. Open ngrok tunnel and print public URL
- **VALIDATE**: Copy into Kaggle notebook, run all cells, test with `curl {ngrok_url}/health`

### Task 8: CREATE app/main.py

- **IMPLEMENT**: Local FastAPI server
- **IMPORTS**: `fastapi`, `httpx`, `app.config`, `app.schemas`
- **ENDPOINTS**:
  - `POST /ocr`: Validate file (type, size), send to Kaggle `/predict`, return OCRResponse
  - `GET /health`: Ping Kaggle `/health`, return HealthResponse
- **GOTCHA**: Use `httpx.AsyncClient` with timeout, handle Kaggle 503/timeout errors gracefully
- **VALIDATE**: `uvicorn app.main:app --port 8080` then `curl -X POST http://localhost:8080/ocr -F "image=@test.jpg"`

### Task 9: CREATE streamlit_app/app.py

- **IMPLEMENT**: Streamlit UI
- **IMPORTS**: `streamlit`, `requests`, `PIL`, `io`, `time`
- **UI ELEMENTS**:
  - `st.title("OCR Tool")`
  - `st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])`
  - `st.text_input("Prompt", value="Extract all text from this image")`
  - `st.button("Extract Text")`
  - `st.image(preview)` — show uploaded image
  - `st.text_area("Result")` — show OCR text
  - `st.code(processing_time)` — show time
- **GOTCHA**: Use `requests` (sync) not `httpx` for Streamlit (Streamlit runs its own event loop)
- **VALIDATE**: `streamlit run streamlit_app/app.py`

### Task 10: CREATE README.md

- **IMPLEMENT**: Setup and usage documentation
- **SECTIONS**:
  1. Overview
  2. Architecture diagram (text-based)
  3. Prerequisites (Kaggle account, ngrok token, Python 3.10+)
  4. Kaggle Setup (steps to create notebook, paste code, get ngrok URL)
  5. Local Setup (pip install, .env, run FastAPI, run Streamlit)
  6. Usage
  7. Troubleshooting
- **VALIDATE**: `cat README.md`

---

## TESTING STRATEGY

### Unit Tests

No unit tests for this project — it's a small tool with thin proxy layers. Validation is done via manual testing and the VALIDATE commands above.

### Integration Tests

Manual end-to-end test:
1. Run Kaggle notebook → get ngrok URL
2. Set ngrok URL in .env
3. Run `uvicorn app.main:app`
4. Run `streamlit run streamlit_app/app.py`
5. Upload image in Streamlit → verify OCR text appears

### Edge Cases

- Upload non-image file → should return 400 error
- Upload image > 10MB → should return 413 error
- Kaggle notebook not running → should return 503 with clear message
- Kaggle request timeout (>120s) → should return 504
- Empty prompt → should use default prompt
- Very large image → model's `max_pixels` setting handles resizing

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
python -m py_compile app/main.py
python -m py_compile app/config.py
python -m py_compile app/schemas.py
python -m py_compile streamlit_app/app.py
python -m py_compile kaggle/notebook.py
```

### Level 2: Import Check

```bash
python -c "from app.main import app; print('FastAPI app loaded')"
python -c "from app.config import KAGGLE_NGROK_URL; print('Config loaded')"
python -c "from app.schemas import OCRResponse; print('Schemas loaded')"
```

### Level 3: Manual Validation

```bash
# Start FastAPI
uvicorn app.main:app --port 8080 &

# Test health endpoint
curl http://localhost:8080/health

# Test OCR endpoint (requires Kaggle to be running)
curl -X POST http://localhost:8080/ocr -F "image=@test_image.jpg" -F "prompt=Extract text"

# Start Streamlit
streamlit run streamlit_app/app.py
```

---

## ACCEPTANCE CRITERIA

- [ ] Kaggle notebook loads Qwen2.5-VL-7B-Instruct and serves predictions via FastAPI
- [ ] ngrok tunnel exposes Kaggle API to public internet
- [ ] Local FastAPI validates file type (jpg/png/bmp/webp) and size (10MB max)
- [ ] Local FastAPI proxies OCR requests to Kaggle with 120s timeout
- [ ] Health endpoint checks Kaggle connectivity
- [ ] Streamlit UI allows image upload with preview
- [ ] Streamlit UI shows OCR result text in copyable text area
- [ ] Streamlit UI shows processing time
- [ ] Error cases return clear messages (400 invalid file, 413 too large, 503 Kaggle down, 504 timeout)
- [ ] README documents full setup process for Kaggle and local

---

## COMPLETION CHECKLIST

- [ ] All 10 tasks completed in order
- [ ] Python syntax check passes for all files
- [ ] Import check passes for all modules
- [ ] FastAPI starts without errors
- [ ] Streamlit starts without errors
- [ ] End-to-end OCR flow works with Kaggle running
- [ ] README is complete and accurate

---

## NOTES

**Key Design Decisions:**
- Using `httpx` (async) for local FastAPI → Kaggle calls because FastAPI is async-native
- Using `requests` (sync) for Streamlit → local FastAPI calls because Streamlit has its own event loop
- Kaggle notebook code is in a single file (`kaggle/notebook.py`) because Kaggle notebooks are typically single-script
- No database — fully stateless
- No auth — developer tool for personal use

**Risks:**
- Kaggle session timeout (~60min idle, ~12h max) — user needs to restart notebook
- ngrok free tier: random URL changes each restart — user must update .env
- T4 GPU memory: 15GB model on 16GB VRAM is tight — `max_pixels` limit is critical
- Model download takes ~5-10 min on first Kaggle run

**Confidence Score: 8/10** — Straightforward architecture, well-documented model usage. Main risk is Kaggle environment variability and first-time model download issues.
