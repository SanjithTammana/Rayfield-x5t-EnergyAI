from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()                                   # reads .env if present

# ---------- File locations ----------
ROOT_DIR   = Path(__file__).parents[1]
MODEL_DIR  = ROOT_DIR / "models"
DB_PATH    = ROOT_DIR / "events.sqlite"

# ---------- API Keys ----------
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
ZAPIER_WEBHOOK_URL  = os.getenv("ZAPIER_WEBHOOK_URL")