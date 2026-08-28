import asyncio
import logging
import tempfile
import threading

from . import config

log = logging.getLogger("diana.stt")

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            log.info("loading whisper model '%s'…", config.STT_MODEL)
            _model = WhisperModel(
                config.STT_MODEL,
                device="cpu",
                compute_type="int8",
                download_root=str(config.MODELS_DIR),
            )
            log.info("whisper ready")
        return _model


def warm():
    """Preload in a background thread so the first voice command is fast."""
    threading.Thread(target=_get_model, daemon=True).start()


def _transcribe_sync(audio_bytes: bytes, suffix: str) -> str:
    model = _get_model()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(audio_bytes)
        f.flush()
        segments, _info = model.transcribe(f.name, beam_size=3, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


async def transcribe(audio_bytes: bytes, suffix: str = ".webm") -> str:
    return await asyncio.to_thread(_transcribe_sync, audio_bytes, suffix)
