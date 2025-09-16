# 🎓 Edubot Backend (AIoT Classroom Assistant)

Edubot là chatbot hỗ trợ giáo viên phân tích hành vi lớp học và gợi ý cải thiện bài giảng, slide, kỹ năng giảng dạy.  
Trả lời **bằng tiếng Việt**, dựa vào RAG (FAISS + dữ liệu scrape UNESCO, Edutopia, Tes, Khan Academy, ISTE, …) và số liệu thực tế từ lớp học.

---

## 🚀 Cài đặt

### Yêu cầu
- Python >= 3.10
- [Ollama](https://ollama.com/download) (để chạy model ngôn ngữ)
- Đã pull model Ollama:
  ```bash
  ollama pull phi3:mini
  ```

### Clone & cài thư viện

```bash
git clone <link-repo>
cd chatbot-edubot
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Cấu hình

Tạo file `.env` từ mẫu:

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/education.index
CHUNKS_PATH=data/education_chunks.pkl
```

### Khởi động server

```bash
uvicorn server:app --host 0.0.0.0 --port 5000
```

Server chạy tại: [http://localhost:5000](http://localhost:5000)

---

## 🔌 API Endpoints

### `GET /health`
Kiểm tra trạng thái:
```json
{"ok": true, "ollama": true}
```

### `POST /chat`
Input mẫu:
```json
{
  "query": "Gợi ý cải thiện mức độ tập trung trong 15 phút đầu tiết học?",
  "k": 4,
  "csv_paths": ["data/detections_20250913_224734.csv"],
  "json_paths": ["data/enhanced_session_stats_20250913_224734.json"]
}
```
Output mẫu:
```json
{"response": "…trả lời bằng tiếng Việt, gồm tóm tắt, kế hoạch hành động, checklist…"}
```

### `POST /ingest`
Tuỳ chọn: rebuild FAISS index từ `education_content.txt`.

---

## 🐳 Chạy bằng Docker

```bash
docker build -t edubot-backend .
docker run -d -p 5000:5000 --env-file .env edubot-backend
```

---

## ✅ Lưu ý bảo mật & dữ liệu

- Đảm bảo Ollama luôn chạy (`ollama serve`).
- Dữ liệu CSV/JSON lớp học phải **ẩn danh**, tuân thủ bảo mật.
- Nếu deploy public: thêm HTTPS, giới hạn truy cập, log & giám sát.

---

🚀 Cách 1 — Chạy trực tiếp bằng Python (dễ nhất)
Bước 1. Cài Ollama

Tải Ollama
 → cài đặt → mở terminal chạy:

ollama pull phi3:mini
ollama serve


Kiểm tra:

curl http://localhost:11434/api/tags


→ phải thấy "phi3:mini".

Bước 2. Chuẩn bị backend

Copy toàn bộ project chatbot-edubot (các file server.py, ai_client.py, rag.py, … và thư mục data/) vào máy tính cá nhân.

Tạo môi trường ảo:

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate


Cài thư viện:

pip install -r requirements.txt

Bước 3. Cấu hình .env

Tạo file .env trong thư mục gốc:

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/education.index
CHUNKS_PATH=data/education_chunks.pkl

Bước 4. Chạy server
uvicorn server:app --host 0.0.0.0 --port 5000


→ Backend chạy tại http://localhost:5000.

Bước 5. Test

Mở Postman, Insomnia hoặc trình duyệt + fetch API:

curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Gợi ý cải thiện mức độ tập trung trong 15 phút đầu tiết học?"}'

🐳 Cách 2 — Chạy bằng Docker (tiện deploy)
Bước 1. Build image

Trong thư mục project:

docker build -t edubot-backend .

Bước 2. Chạy container
docker run -d -p 5000:5000 --env-file .env edubot-backend

Bước 3. Kiểm tra

Mở http://localhost:5000/health
 → thấy:

{"ok": true, "ollama": true}

🌐 Tích hợp với UI

Nếu frontend nằm chung máy: chỉ cần gọi API:

fetch("http://localhost:5000/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({ query: "Gợi ý tăng tương tác học sinh" })
}).then(r => r.json()).then(console.log)


Nếu muốn chạy trên domain/webserver: thêm Nginx proxy để trỏ /api → backend.

👉 Tóm lại:

Máy cá nhân: cài Ollama + Python, chạy uvicorn server:app.
Test: gọi http://localhost:5000/chat từ UI hoặc curl.
Triển khai: nếu muốn gọn gàng → dùng Docker