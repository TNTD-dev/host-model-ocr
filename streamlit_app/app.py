import time

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FASTAPI_URL = "http://localhost:8080"
DEFAULT_PROMPT = "Extract all text from this image"

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🔍 OCR Tool")
st.caption("Extract text from images using Qwen2.5-VL-7B-Instruct")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
    )

    prompt = st.text_input(
        "Prompt (optional)",
        value=DEFAULT_PROMPT,
        help="Custom prompt for the model",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

with col2:
    if uploaded_file:
        if st.button("🔍 Extract Text", type="primary", use_container_width=True):
            with st.spinner("Processing... This may take 30-120 seconds on first run."):
                start_time = time.time()

                try:
                    uploaded_file.seek(0)
                    files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"prompt": prompt}

                    resp = requests.post(
                        f"{FASTAPI_URL}/ocr",
                        files=files,
                        data=data,
                        timeout=180,
                    )

                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"✅ Done in {result['processing_time']}s")

                        st.text_area(
                            "OCR Result",
                            value=result["text"],
                            height=400,
                            help="Copy the extracted text below",
                        )

                        st.code(result["text"], language=None)
                    else:
                        error_detail = resp.json().get("detail", resp.text)
                        st.error(f"Error {resp.status_code}: {error_detail}")

                except requests.ConnectionError:
                    st.error(
                        "❌ Cannot connect to FastAPI server. "
                        "Make sure it's running: `uvicorn app.main:app --port 8080`"
                    )
                except requests.Timeout:
                    st.error("❌ Request timed out. The model may still be loading on Kaggle.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
    else:
        st.info("👈 Upload an image to get started")

# ---------------------------------------------------------------------------
# Sidebar: Health Check
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Status")

    if st.button("Check Server Health"):
        try:
            resp = requests.get(f"{FASTAPI_URL}/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"FastAPI: {data['status']}")
                if data.get("kaggle_reachable"):
                    st.success("Kaggle: reachable")
                else:
                    st.warning("Kaggle: not reachable")
            else:
                st.error(f"FastAPI: error {resp.status_code}")
        except requests.ConnectionError:
            st.error("FastAPI: not running")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.markdown(
        "**Setup:**\n"
        "1. Run Kaggle notebook\n"
        "2. Copy ngrok URL to `.env`\n"
        "3. `uvicorn app.main:app --port 8080`\n"
        "4. `streamlit run streamlit_app/app.py`"
    )
