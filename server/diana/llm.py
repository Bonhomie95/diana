import json
import logging
import re

import httpx

from . import config

log = logging.getLogger("diana.llm")

_client: httpx.AsyncClient | None = None
_base_url: str | None = None
_think_capable: dict[str, bool] = {}


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=900.0, write=60.0))
    return _client


async def base_url(force: bool = False) -> str | None:
    """Probe candidate Ollama endpoints, cache the first that answers."""
    global _base_url
    if _base_url and not force:
        return _base_url
    for url in config.OLLAMA_CANDIDATES:
        try:
            r = await client().get(f"{url}/api/tags", timeout=4.0)
            if r.status_code == 200:
                _base_url = url
                log.info("Ollama found at %s", url)
                for m in r.json().get("models", []):
                    _think_capable[m["name"]] = "thinking" in (m.get("capabilities") or [])
                return url
        except Exception:
            continue
    _base_url = None
    return None


async def list_models() -> list[dict]:
    url = await base_url()
    if not url:
        return []
    try:
        r = await client().get(f"{url}/api/tags")
        models = r.json().get("models", [])
        for m in models:
            _think_capable[m["name"]] = "thinking" in (m.get("capabilities") or [])
        return models
    except Exception:
        return []


async def has_model(name: str) -> bool:
    models = await list_models()
    names = {m["name"] for m in models} | {m["name"].split(":")[0] for m in models}
    return name in names or name.split(":")[0] in names


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


async def chat(messages: list[dict], model: str | None = None,
               temperature: float = 0.7, num_ctx: int = 16384) -> str:
    """Non-streaming chat completion. Raises RuntimeError when Ollama is unreachable."""
    url = await base_url()
    if not url:
        url = await base_url(force=True)
    if not url:
        raise RuntimeError("Ollama is unreachable. Is it running?")
    model = model or config.DIANA_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    # Disable chain-of-thought for snappy replies on thinking-capable models.
    if _think_capable.get(model):
        payload["think"] = False
    try:
        r = await client().post(f"{url}/api/chat", json=payload)
        if r.status_code != 200 and "think" in payload:
            payload.pop("think")
            r = await client().post(f"{url}/api/chat", json=payload)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        return strip_think(content)
    except httpx.HTTPError as e:
        global _base_url
        _base_url = None
        raise RuntimeError(f"LLM call failed: {e}") from e


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of model output, tolerantly."""
    text = strip_think(text)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:i + 1])
                        break
        next_start = text.find("{", start + 1)
        if candidates:
            break
        start = next_start
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            # common repair: trailing commas
            try:
                obj = json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


async def pull_model(name: str, progress_cb=None):
    url = await base_url()
    if not url:
        raise RuntimeError("Ollama is unreachable")
    async with client().stream("POST", f"{url}/api/pull",
                               json={"model": name}, timeout=None) as r:
        async for line in r.aiter_lines():
            if not line.strip():
                continue
            try:
                st = json.loads(line)
            except Exception:
                continue
            if progress_cb:
                total, done = st.get("total"), st.get("completed")
                pct = round(done / total * 100) if total and done else None
                await progress_cb(st.get("status", ""), pct)
