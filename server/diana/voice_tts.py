import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from . import config

log = logging.getLogger("diana.tts")

PIPER_VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

_voice_ready = False
_voice_failed = False
_dl_lock = asyncio.Lock()


def _voice_paths() -> tuple[Path, Path]:
    v = config.TTS_VOICE  # e.g. en_US-lessac-medium
    return (config.VOICES_DIR / f"{v}.onnx", config.VOICES_DIR / f"{v}.onnx.json")


def _voice_urls() -> tuple[str, str]:
    v = config.TTS_VOICE
    m = re.match(r"([a-z]+)_([A-Z]+)-([a-z0-9_]+)-([a-z_]+)", v)
    if not m:
        raise ValueError(f"unrecognized piper voice name: {v}")
    lang, region, name, quality = m.groups()
    base = f"{PIPER_VOICE_BASE}/{lang}/{lang}_{region}/{name}/{quality}/{v}"
    return f"{base}.onnx", f"{base}.onnx.json"


async def _ensure_piper_voice() -> bool:
    global _voice_ready, _voice_failed
    if _voice_ready:
        return True
    if _voice_failed:
        return False
    async with _dl_lock:
        if _voice_ready:
            return True
        onnx, meta = _voice_paths()
        if onnx.exists() and meta.exists():
            _voice_ready = True
            return True
        try:
            config.ensure_dirs()
            onnx_url, meta_url = _voice_urls()
            log.info("downloading piper voice %s…", config.TTS_VOICE)
            async with httpx.AsyncClient(timeout=None, follow_redirects=True) as c:
                for url, dest in ((onnx_url, onnx), (meta_url, meta)):
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    async with c.stream("GET", url) as r:
                        r.raise_for_status()
                        with open(tmp, "wb") as f:
                            async for chunk in r.aiter_bytes(1 << 16):
                                f.write(chunk)
                    tmp.rename(dest)
            _voice_ready = True
            log.info("piper voice ready")
            return True
        except Exception as e:
            log.warning("piper voice download failed: %s", e)
            _voice_failed = True
            return False


def _piper_sync(text: str) -> bytes | None:
    exe = shutil.which("piper")
    if not exe:
        return None
    onnx, _ = _voice_paths()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as out:
        p = subprocess.run(
            [exe, "--model", str(onnx), "--output_file", out.name],
            input=text.encode("utf-8"),
            capture_output=True, timeout=120,
        )
        if p.returncode != 0:
            log.warning("piper failed: %s", p.stderr[-400:].decode(errors="replace"))
            return None
        data = Path(out.name).read_bytes()
        return data if len(data) > 128 else None


async def _edge_tts(text: str) -> bytes | None:
    try:
        import edge_tts
        buf = bytearray()
        communicate = edge_tts.Communicate(text, config.EDGE_VOICE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf) or None
    except Exception as e:
        log.warning("edge-tts failed: %s", e)
        return None


def clean_for_speech(text: str) -> str:
    """Strip markdown so TTS doesn't read symbols aloud."""
    t = re.sub(r"```.*?```", " Code block omitted. ", text, flags=re.DOTALL)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"[*_>|#]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:1200]


async def synthesize(text: str) -> tuple[bytes, str] | None:
    """Returns (audio_bytes, mime) or None — the UI then falls back to browser TTS."""
    text = clean_for_speech(text)
    if not text or config.TTS_ENGINE == "off":
        return None
    if config.TTS_ENGINE == "piper" and await _ensure_piper_voice():
        audio = await asyncio.to_thread(_piper_sync, text)
        if audio:
            return audio, "audio/wav"
    audio = await _edge_tts(text)
    if audio:
        return audio, "audio/mpeg"
    return None
