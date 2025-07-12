# EduSenseAI 🚀
AI + IoT system using Intel UP4000 and camera to analyze student behavior and optimize teaching curriculum.

## 🧠 Mục tiêu dự án
- Sử dụng camera trong lớp học để phân tích hành vi học tập (chú ý, giơ tay, viết bài,...).
- Kết hợp nội dung bài giảng và kết quả kiểm tra để đề xuất cải tiến giáo án.
- Chạy mô hình AI tại thiết bị biên UP4000 (Edge AI).

## 🛠 Công nghệ sử dụng
- **Ngôn ngữ:** Python
- **AI Framework:** OpenVINO, YOLOv5, scikit-learn
- **IoT phần cứng:** Intel UP4000
- **Giao tiếp:** Rule-based chatbot hoặc form feedback

## 👨‍👩‍👧‍👦 Phân chia nhóm
| Thành viên | Vai trò chính                                    |
|---------------------------|-----------------------------------|
| Tran Huu Hoang Long       | AI modeling, training, inference  |
| Nguyen Khanh Minh         | Cài đặt UP4000, camera, test edge |
| Tran Huu Hoang Long       | Annotate data, xử lý bài giảng    |
| Tran Huu Hoang Long       | Demo + báo cáo + giao diện        |

## 📅 Timeline chính
| Ngày        | Nội dung                        |
|-------------|---------------------------------|
| 01–11/07    | Tìm + chuẩn hóa dữ liệu         |
| 12–17/07    | Huấn luyện model, sync nội dung |
| 18–20/07    | Demo UP4000 + chatbot + báo cáo |

## 🔧 Hướng dẫn chạy
```bash
# Clone repo
git clone https://github.com/your-username/EduSenseAI.git
cd EduSenseAI

# Cài thư viện
pip install -r requirements.txt

# Chạy script tách frame video
python src/data/preprocess_video.py --input dataset/raw/video1.mp4

# Huấn luyện mô hình
python src/models/train_model.py

# Test trên webcam
python src/models/infer.py


EduSenseAI/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   ├── raw/                  # Video, hình ảnh gốc
│   ├── annotations/          # Nhãn hành vi (COCO/VOC)
│   ├── slides/               # Slide bài giảng (PDF, PPTX)
│   ├── transcripts/          # Chuyển giọng nói thành văn bản
│   └── processed/            # Dữ liệu đã chuẩn hóa, dùng huấn luyện
├── notebooks/
│   ├── 01_annotation_preview.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data/
│   │   ├── convert_pdf.py           # Chuyển slide PDF → text
│   │   ├── extract_audio.py         # Tách âm thanh từ video
│   │   └── preprocess_video.py      # Tách frame, resize ảnh
│   ├── models/
│   │   ├── train_model.py           # Huấn luyện YOLO hoặc model OpenVINO
│   │   └── infer.py                 # Inference từ webcam/video
│   ├── sync/
│   │   └── sync_content_video.py    # Đồng bộ bài giảng + video
│   └── chatbot/
│       └── rule_based_bot.py        # Gợi ý cải thiện giáo trình
├── up4000_deploy/
│   ├── openvino_ir/                 # Mô hình IR để chạy trên UP4000
│   └── deploy_script.py             # Script demo camera UP4000
├── results/
│   ├── logs/
│   ├── figures/
│   └── report.md
├── LICENSE
└── CONTRIBUTING.md
