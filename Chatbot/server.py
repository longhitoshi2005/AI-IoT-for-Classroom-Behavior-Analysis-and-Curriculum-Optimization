import os
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


from prompts import SYSTEM_PROMPT, USER_TEMPLATE
from ai_client import chat as ollama_chat, ollama_health, OllamaError
from rag import load_assets, build_context


PORT = int(os.getenv("PORT", 5000))


app = FastAPI(title="Edubot Backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    query: str
    k: int = 3
    csv_paths: Optional[List[str]] = []
    json_paths: Optional[List[str]] = []


@app.on_event("startup")
def _warmup():
    load_assets()

@app.get("/health")
def health():
    return {"ok": True, "ollama": ollama_health()}


@app.post("/chat")
def chat(inb: ChatIn):
    q = (inb.query or "").strip()
    if not q:
        return {"response": "Vui lòng nhập câu hỏi hợp lệ."}
    context = build_context(q, inb.k, inb.csv_paths or [], inb.json_paths or [])


    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(context=context, query=q)},
    ]
    try:
        out = ollama_chat(messages).strip()
        if not out:
            out = "Xin lỗi, tôi chưa tạo được câu trả lời. Bạn thử hỏi lại bằng cách nêu mục tiêu dạy học cụ thể nhé."
        return {"response": out}
    except OllamaError as e:
        return {"response": f"Không kết nối được tới mô hình AI. Hãy kiểm tra Ollama (URL, model, trạng thái dịch vụ). Chi tiết: {e}"}


# tuỳ chọn: endpoint rebuild index nếu bạn muốn làm mới dữ liệu sau này
from rag import INDEX_PATH, CHUNKS_PATH
@app.post("/ingest")
def ingest():
    # Bạn có thể gọi embed_build.py bằng subprocess, hoặc viết lại logic tại đây.
    return {"status": "ok", "note": "Chạy embed_build.py để cập nhật FAISS khi education_content.txt thay đổi."}


# chạy: uvicorn server:app --host 0.0.0.0 --port 5000
