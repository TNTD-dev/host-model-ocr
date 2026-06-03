import os
from dotenv import load_dotenv

load_dotenv()

KAGGLE_NGROK_URL = os.getenv("KAGGLE_NGROK_URL", "")
KAGGLE_TIMEOUT = int(os.getenv("KAGGLE_TIMEOUT", "120"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
