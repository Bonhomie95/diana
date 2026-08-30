import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "diana.db"
VOICES_DIR = DATA_DIR / "voices"
MODELS_DIR = DATA_DIR / "models"
SKILLS_DIR = DATA_DIR / "skills"

# LLM
OLLAMA_URL = os.environ.get("OLLAMA_URL", "").strip()
OLLAMA_CANDIDATES = [
    u for u in [
        OLLAMA_URL,
        "http://host.docker.internal:11434",
        "http://ollama:11434",
        "http://localhost:11434",
    ] if u
]
DIANA_MODEL = os.environ.get("DIANA_MODEL", "qwen3.6:latest")
WORKER_MODEL = os.environ.get("WORKER_MODEL", "mistral:latest")

# Voice
STT_MODEL = os.environ.get("STT_MODEL", "base")
TTS_ENGINE = os.environ.get("TTS_ENGINE", "piper")  # piper | edge | off
TTS_VOICE = os.environ.get("TTS_VOICE", "en_US-lessac-medium")
EDGE_VOICE = os.environ.get("EDGE_VOICE", "en-US-AriaNeural")

# Security: when set, every request must present this token once
# (open http://…:8080/?token=<value> — a cookie keeps you signed in).
DIANA_TOKEN = os.environ.get("DIANA_TOKEN", "").strip()

LAN_IP = os.environ.get("DIANA_LAN_IP", "").strip()

# Runtime
PORT = int(os.environ.get("PORT", "8080"))
TLS_PORT = int(os.environ.get("TLS_PORT", "8443"))
MAX_TASK_ATTEMPTS = int(os.environ.get("MAX_TASK_ATTEMPTS", "2"))
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "2"))
CHAT_HISTORY = int(os.environ.get("CHAT_HISTORY", "14"))


def diana_model() -> str:
    from . import db
    return db.get_setting("DIANA_MODEL", DIANA_MODEL)


def worker_model() -> str:
    from . import db
    return db.get_setting("WORKER_MODEL", WORKER_MODEL)


def ensure_dirs():
    for d in (DATA_DIR, VOICES_DIR, MODELS_DIR, SKILLS_DIR):
        d.mkdir(parents=True, exist_ok=True)
