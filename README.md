# EduSenseAI - Intel Classroom Behavior Analysis 🚀

**AI + IoT system using Intel UP4000 and YOLOv8 to analyze student behavior and optimize teaching curriculum.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2023.0-blue.svg)](https://docs.openvino.ai/)
[![Intel UP4000](https://img.shields.io/badge/Intel-UP4000-lightblue.svg)](https://www.intel.com/)

## 🎯 Project Overview / Mục tiêu dự án

**English:**
Real-time student behavior detection system using computer vision to analyze classroom engagement:
- **handraise**: Student raising hand for participation
- **write**: Student writing or taking notes
- **read**: Student reading materials

**Tiếng Việt:**
- Sử dụng camera trong lớp học để phân tích hành vi học tập (chú ý, giơ tay, viết bài,...)
- Kết hợp nội dung bài giảng và kết quả kiểm tra để đề xuất cải tiến giáo án
- Chạy mô hình AI tại thiết bị biên UP4000 (Edge AI)

## 🏗️ Technology Stack / Công nghệ sử dụng

| Component | Technology |
|-----------|------------|
| **AI Framework** | YOLOv8 + OpenVINO |
| **Language** | Python 3.8+ |
| **Hardware** | Intel UP4000 Edge Device |
| **Computer Vision** | OpenCV, Ultralytics |
| **Deployment** | OpenVINO Runtime |
| **Interface** | Rule-based chatbot, Web dashboard |

## 👥 Team Structure / Phân chia nhóm

| Team Member | Primary Role | Responsibilities |
|-------------|-------------|------------------|
| **Tran Huu Hoang Long** | AI Engineer | Model training, inference, data annotation |
| **Nguyen Khanh Minh** | Edge Deployment | UP4000 setup, camera integration, edge testing |

## 📅 Project Timeline / Timeline chính

| Date / Ngày | Milestone / Nội dung |
|-------------|---------------------|
| **01-11/07** | Data Collection & Preparation / Tìm + chuẩn hóa dữ liệu |
| **12-17/07** | Model Training & Content Sync / Huấn luyện model, sync nội dung |
| **18-20/07** | UP4000 Demo & Reporting / Demo UP4000 + chatbot + báo cáo |

## 📁 Project Structure / Cấu trúc dự án

```
EduSenseAI/
├── README.md
├── requirements.txt
├── .gitignore
├── data.yaml                        # YOLO dataset configuration
├── dataset/                         # ⚠️ NOT IN GIT (download separately)
│   ├── raw/                         # Original videos, images
│   ├── annotations/                 # Behavior labels (YOLO format)
│   ├── slides/                      # Lecture slides (PDF, PPTX)
│   ├── transcripts/                 # Speech-to-text transcriptions
│   └── processed/                   # Processed data for training
│       ├── images/
│       │   ├── train/               # Training images (5,193 files)
│       │   └── val/                 # Validation images (1,671 files)
│       └── labels/
│           ├── train/               # Training labels (YOLO format)
│           └── val/                 # Validation labels
├── notebooks/
│   ├── 01_annotation_preview.ipynb  # Dataset visualization
│   ├── 02_preprocessing.ipynb       # Data preprocessing
│   └── 03_model_training.ipynb      # Training analysis
├── src/
│   ├── data/
│   │   ├── convert_pdf.py           # PDF slides → text conversion
│   │   ├── extract_audio.py         # Audio extraction from video
│   │   └── preprocess_video.py      # Frame extraction, image resize
│   ├── models/
│   │   ├── train_model.py           # YOLOv8 training script
│   │   └── infer.py                 # OpenVINO inference script
│   ├── sync/
│   │   └── sync_content_video.py    # Lecture-video synchronization
│   └── chatbot/
│       └── rule_based_bot.py        # Curriculum improvement suggestions
├── up4000_deploy/
│   ├── openvino_ir/                 # ⚠️ NOT IN GIT (generated during training)
│   └── deploy_script.py             # Real-time camera demo
├── results/                         # ⚠️ NOT IN GIT (generated during training)
│   ├── train/                       # Training results & metrics
│   ├── logs/                        # Training logs
│   ├── figures/                     # Visualization plots
│   └── report.md                    # Performance report
├── venv/                            # ⚠️ NOT IN GIT (virtual environment)
└── docs/
    ├── DATASET.md                   # Dataset download instructions
    ├── CONTRIBUTING.md
    └── DEPLOYMENT.md
```

**⚠️ Important Notes / Lưu ý quan trọng:**
- `dataset/`, `results/`, `venv/`, and model files are **NOT included in Git** due to size limitations
- Dataset: ~1.4GB (6,864 images) - Download separately 
- Virtual environment: Recreate using `requirements.txt`
- Model weights: Generated during training process

## 📊 Dataset Statistics / Thống kê dữ liệu

| Metric | Value |
|--------|-------|
| **Total Images** | 6,864 |
| **Training Set** | 5,193 images |
| **Validation Set** | 1,671 images |
| **Classes** | 3 (handraise, write, read) |
| **Dataset Size** | ~1.4 GB |
| **Git Status** | ❌ Excluded (too large) |

### Class Distribution / Phân bố lớp:
- **write**: ~17,000 instances (52%) 📝
- **handraise**: ~10,500 instances (32%) 🖐️
- **read**: ~6,500 instances (16%) 📖

## 🚀 Quick Start / Hướng dẫn chạy nhanh

### 1. Clone Repository / Clone repo
```bash
git clone https://github.com/your-username/EduSenseAI.git
cd EduSenseAI
```

### 2. Setup Environment / Thiết lập môi trường
```bash
# Create virtual environment (NOT in Git)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Dataset / Tải dataset
```bash
# Dataset is NOT in Git repository
# Contact project maintainer for dataset access
# Extract dataset to ./dataset/ folder
# Verify structure:
python -c "
import os
if os.path.exists('dataset/processed/images/train'):
    train_count = len([f for f in os.listdir('dataset/processed/images/train') if f.endswith('.jpg')])
    val_count = len([f for f in os.listdir('dataset/processed/images/val') if f.endswith('.jpg')])
    print(f'✅ Dataset found: {train_count} train, {val_count} val images')
else:
    print('❌ Dataset not found. Please download and extract to ./dataset/')
"
```

### 4. Train Model / Huấn luyện mô hình
```bash
# Train YOLOv8 model (results/ folder will be created)
python src/models/train_model.py

# Monitor training progress
tensorboard --logdir results/train
```

### 5. Test Inference / Test suy luận
```bash
# Test on webcam
python src/models/infer.py

# Test on UP4000 device
python up4000_deploy/deploy_script.py
```

## 🎯 Model Performance / Hiệu suất mô hình

| Metric | Value |
|--------|-------|
| **Architecture** | YOLOv8s (Small) |
| **Input Size** | 640×640 pixels |
| **Inference Speed** | ~30 FPS on Intel UP4000 |
| **Model Size** | ~22 MB (OpenVINO IR) |
| **Precision (mAP@0.5)** | ~85% (estimated) |

### Detection Classes / Lớp phát hiện:
```yaml
names:
  0: handraise  # Student raising hand / Học sinh giơ tay
  1: write      # Student writing / Học sinh viết bài  
  2: read       # Student reading / Học sinh đọc sách
```

## 🎨 Visualization / Trực quan hóa

Detection boxes are color-coded / Hộp phát hiện được mã hóa màu:
- 🟢 **handraise**: Green / Xanh lá
- 🔵 **write**: Blue / Xanh dương
- 🔴 **read**: Red / Đỏ

## 🛠️ Hardware Requirements / Yêu cầu phần cứng

### Development Environment / Môi trường phát triển:
- **CPU**: Intel Core i5+ or equivalent
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB free space (including dataset)
- **GPU**: Optional (CUDA-compatible for faster training)

### Deployment (Intel UP4000) / Triển khai:
- **Processor**: Intel Atom x7-E3950 Quad Core
- **RAM**: 4GB LPDDR4
- **Storage**: 64GB eMMC
- **Camera**: USB 3.0 or MIPI camera
- **OS**: Ubuntu 20.04 LTS

## 📈 Training Configuration / Cấu hình huấn luyện

```python
# Training parameters
EPOCHS = 50
BATCH_SIZE = 16  
IMAGE_SIZE = 640
LEARNING_RATE = 0.001
DEVICE = 'cpu'  # or 'cuda' if available

# Class weights (to handle imbalance)
CLASS_WEIGHTS = [0.8, 0.5, 1.5]  # handraise, write, read
```

## 🔄 CI/CD Pipeline / Quy trình tự động

```bash
# Automated workflow
1. Environment setup → python -m venv venv && pip install -r requirements.txt
2. Dataset validation → python src/data/validate.py
3. Model training → python src/models/train_model.py  
4. Performance testing → python src/models/evaluate.py
5. OpenVINO conversion → Auto-export during training
6. Edge deployment → python up4000_deploy/deploy_script.py
```

## 📊 Results & Analytics / Kết quả & Phân tích

Training generates (in `results/` folder, NOT in Git):
- `labels.jpg` - Ground truth annotations / Nhãn thực tế
- `train_batch.jpg` - Training batch samples / Mẫu batch huấn luyện
- `results.png` - Loss curves and metrics / Đường cong loss và metrics
- Model weights in `results/train/handraise_write_read_detection/weights/`

## 🤖 Edge AI Deployment / Triển khai Edge AI

### OpenVINO Optimization / Tối ưu hóa OpenVINO:
```bash
# Automatic conversion during training
model.export(format='openvino', optimize=True)

# Generated files (NOT in Git):
# - up4000_deploy/openvino_ir/best.xml
# - up4000_deploy/openvino_ir/best.bin
```

### UP4000 Performance / Hiệu suất UP4000:
- **Inference Speed**: 30+ FPS
- **Power Consumption**: ~10W
- **Memory Usage**: <2GB RAM
- **Real-time Processing**: ✅ Supported

## 📝 API Documentation / Tài liệu API

### Inference API:
```python
from src.models.infer import OpenVINOInference

# Initialize model (requires trained model files)
detector = OpenVINOInference('up4000_deploy/openvino_ir/best.xml')

# Run detection
detections, inference_time = detector.run_inference(image)

# Draw results
result_image = detector.draw_detections(image, detections)
```

## 📦 Repository Size Management / Quản lý kích thước repository

| Component | Size | Git Status | Notes |
|-----------|------|------------|-------|
| **Source Code** | ~5 MB | ✅ Included | Python scripts, configs |
| **Documentation** | ~1 MB | ✅ Included | README, docs |
| **Dataset** | ~1.4 GB | ❌ Excluded | Too large for Git |
| **Virtual Environment** | ~200 MB | ❌ Excluded | Recreate with requirements.txt |
| **Model Weights** | ~50 MB | ❌ Excluded | Generated during training |
| **Training Results** | ~100 MB | ❌ Excluded | Generated during training |

**Total Git Repository**: ~10 MB (clean and fast!)

## 🔒 Security & Privacy / Bảo mật & Quyền riêng tư

- **Data Privacy**: All processing done on-device (no cloud upload)
- **Student Privacy**: Faces are not stored or transmitted
- **GDPR Compliant**: Only behavioral patterns analyzed
- **Local Storage**: All data remains on UP4000 device

## 🤝 Contributing / Đóng góp

1. Fork the repository / Fork repository
2. Create feature branch / Tạo nhánh tính năng: `git checkout -b feature/amazing-feature`
3. **Note**: Don't commit dataset, venv, or large files
4. Commit changes / Commit thay đổi: `git commit -m 'Add amazing feature'`
5. Push to branch / Push lên nhánh: `git push origin feature/amazing-feature`
6. Open Pull Request / Mở Pull Request

## 📞 Contact & Support / Liên hệ & Hỗ trợ

- **Project Lead**: Tran Huu Hoang Long
- **Hardware Lead**: Nguyen Khanh Minh  
- **Email**: [project-email@example.com]
- **Intel AI Training Program**: 2024

## 📄 License / Giấy phép

This project is developed for Intel AI Training purposes. See `LICENSE` file for details.

## 🙏 Acknowledgments / Lời cảm ơn

- Intel Corporation for UP4000 hardware support
- Ultralytics for YOLOv8 framework
- OpenVINO team for optimization tools
- Intel AI Training Program instructors

---

**Made with ❤️ for intelligent education / Được tạo với ❤️ cho giáo dục thông minh**

*Repository optimized for Git: ~10MB clean codebase, large files excluded via .gitignore*