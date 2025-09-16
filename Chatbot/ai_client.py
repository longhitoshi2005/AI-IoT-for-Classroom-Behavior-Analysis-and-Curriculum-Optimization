import os, time, requests
from typing import List, Dict, Any


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b") # có thể đổi ở .env
TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT", 45))
RETRIES = int(os.getenv("OLLAMA_RETRIES", 2))


class OllamaError(RuntimeError):
    pass


def _chat_payload(messages: List[Dict[str,str]]) -> Dict[str, Any]:
    return {"model": OLLAMA_MODEL, "messages": messages, "stream": False,
    "options": {"temperature": 0.2}}


def ollama_health() -> bool:
    try:
        # tags endpoint: list models
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return True
    except Exception:
        return False


def chat(messages: List[Dict[str,str]]) -> str:
    last_exc = None
    for _ in range(RETRIES + 1):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=_chat_payload(messages), timeout=TIMEOUT_S)
            r.raise_for_status()
            data = r.json()
            # Ollama /api/chat returns {message:{content:...}} or {response:...} depending on version
            if "message" in data and isinstance(data["message"], dict):
                return data["message"].get("content", "").strip()
            return data.get("response", "").strip()
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
    raise OllamaError(str(last_exc))