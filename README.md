# OCR Tool - Qwen2.5-VL-7B-Instruct

Extract text from images using the Qwen2.5-VL-7B-Instruct vision-language model hosted on Kaggle GPU.

## Architecture

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Streamlit UI   │────▶│  FastAPI (local)     │────▶│  Kaggle Notebook     │
│  localhost:8501  │     │  localhost:8080      │     │  FastAPI + ngrok     │
│                 │     │  Proxy + Validation  │     │  Qwen2.5-VL-7B      │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
```

## Prerequisites

- Python 3.10+
- [Kaggle account](https://www.kaggle.com/) with GPU access
- [ngrok account](https://ngrok.com/) (free tier)

## Setup

### Step 1: Kaggle Notebook Setup

1. Go to [Kaggle](https://www.kaggle.com/) and create a new notebook
2. Enable GPU: **Settings** → **Accelerator** → **GPU T4 x2**
3. Copy the contents of `kaggle/notebook.py` into the notebook
4. In the notebook, replace `YOUR_NGROK_TOKEN_HERE` with your ngrok auth token:
   - Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken
5. Run all cells
6. Copy the **Public ngrok URL** printed at the end (e.g., `https://xxxx.ngrok-free.app`)

### Step 2: Local Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd host-model-ocr

# Install dependencies
pip install -r requirements-local.txt

# Create .env file
cp .env.example .env
```

Edit `.env` and paste your ngrok URL:
```
KAGGLE_NGROK_URL=https://xxxx.ngrok-free.app
```

### Step 3: Run

**Terminal 1 - Start FastAPI server:**
```bash
uvicorn app.main:app --port 8080
```

**Terminal 2 - Start Streamlit UI:**
```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. Upload an image (JPG, PNG, BMP, WEBP)
2. Optionally customize the prompt
3. Click "Extract Text"
4. Wait 30-120 seconds (first run is slower due to model loading)
5. Copy the extracted text

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot connect to Kaggle" | Make sure the Kaggle notebook is running |
| "KAGGLE_NGROK_URL not configured" | Set the URL in `.env` file |
| "Request timed out" | Model may be loading; wait and retry |
| "File too large" | Max file size is 10MB |
| Kaggle session ended | Restart the notebook and update ngrok URL in `.env` |

## Project Structure

```
host-model-ocr/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server (local)
│   ├── config.py             # Settings
│   └── schemas.py            # Pydantic models
├── streamlit_app/
│   └── app.py                # Streamlit UI
├── kaggle/
│   └── notebook.py           # Kaggle notebook code
├── requirements-local.txt    # Local dependencies
├── requirements-kaggle.txt   # Kaggle dependencies
├── .env.example              # Environment template
└── README.md
```

## API Endpoints

### Local FastAPI (localhost:8080)

- `POST /ocr` - Upload image, get OCR text
- `GET /health` - Check server and Kaggle connectivity

### Kaggle Notebook (ngrok URL)

- `POST /predict` - Direct model inference
- `GET /health` - Model health check
