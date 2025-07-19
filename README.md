# YOLOv8 Classroom Behavior Detection System

AI-powered classroom behavior analysis system using YOLOv8 and computer vision to analyze student engagement patterns in real-time. This system combines Intel UP4000 edge computing with advanced object detection to monitor classroom activities and optimize teaching effectiveness.

## Project Overview

**Key Behaviors Detected:**
- Hand Raise: Student raising hand for participation and questions
- Writing: Student writing or taking notes during lessons  
- Reading: Student reading materials and textbooks

**Educational Impact:**
- Real-time Engagement Analysis: Monitor student participation patterns
- Curriculum Optimization: Data-driven insights for teaching improvement
- Edge AI Deployment: Privacy-focused on-device processing
- Automated Reporting: Generate engagement analytics for educators

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| AI Framework | YOLOv8 + OpenVINO | Object detection and optimization |
| Language | Python 3.8+ | Core development language |
| Hardware | Intel UP4000 Edge Device | Edge AI deployment platform |
| Computer Vision | OpenCV, Ultralytics | Image processing and ML framework |
| Deployment | OpenVINO Runtime | Intel hardware acceleration |
| Camera Support | Multi-platform | PC webcam, iPhone, Android tablet |

## Development Team

| Team Member | Role | Responsibilities |
|-------------|------|------------------|
| Tran Huu Hoang Long | AI Engineer | Model training, optimization, data annotation |
| Nguyen Khanh Minh | Edge Deployment | UP4000 setup, camera integration, testing |

## Key Features

- High Performance: 92.8 FPS inference speed with 69.6% mAP50 accuracy
- Multi-Camera Support: PC webcam, iPhone (Camo), Android tablet (IP Webcam)
- GPU Acceleration: NVIDIA CUDA support for 20x faster training
- Edge Deployment: Intel UP4000 optimization with OpenVINO
- Real-time Detection: Live video stream processing with bounding boxes
- Privacy-First: All processing done on-device, no cloud dependency

## Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| mAP50 | 69.6% | Mean Average Precision at IoU=0.5 |
| Precision | 64.5% | Overall detection precision |
| Recall | 67.8% | Overall detection recall |
| mAP50-95 | 53.3% | Mean Average Precision IoU=0.5:0.95 |
| Inference Speed | 10.8ms per frame | Average processing time |
| Training Speedup | 20x faster | GPU vs CPU training time |
| Dataset Size | 6,864 images | 5,193 train + 1,671 validation |

## Quick Start

### Option 1: Automated Setup
```bash
git clone https://github.com/longhitoshi2005/AI-IoT-for-Classroom-Behavior-Analysis-and-Curriculum-Optimization.git
cd AI-IoT-for-Classroom-Behavior-Analysis-and-Curriculum-Optimization
python setup.py
```

### Option 2: Manual Setup
```bash
# Clone Repository
git clone https://github.com/longhitoshi2005/AI-IoT-for-Classroom-Behavior-Analysis-and-Curriculum-Optimization.git
cd AI-IoT-for-Classroom-Behavior-Analysis-and-Curriculum-Optimization

# Create Virtual Environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install Dependencies
pip install -r requirements.txt

# GPU Support (Optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## System Requirements

**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- OpenCV-compatible camera
- Windows 10/11, Linux, or macOS

**Recommended for Training:**
- NVIDIA GPU with CUDA support (RTX 3060 or better)
- 16GB RAM
- 50GB free disk space

## Camera Testing Options

### iPhone with Camo (Best Performance)
```bash
python up4000_deploy/deploy_camo_iphone.py
```
- Performance: HD quality (1280x720), 92.8 FPS
- Requirements: iPhone with Camo app + USB connection

### PC Webcam
```bash
python up4000_deploy/deploy_script.py
```
- Standard webcam support
- Automatic camera detection

## Intel UP4000 Deployment

```bash
# Convert model to OpenVINO format
yolo export model=results/train/handraise_write_read_detection/weights/best.pt format=openvino

# Deploy on UP4000
python up4000_deploy/deploy_script.py
```

**Expected UP4000 Performance:**
- Inference Speed: 30-50ms per frame
- FPS: 15-25 FPS
- Power Usage: ~10W

## Training Your Own Model

```bash
# Start training
python src/models/train_model.py

# GPU-accelerated training
python src/models/resume_training_gpu.py
```

## Project Structure

```
AI-IoT-for-Classroom-Behavior-Analysis-and-Curriculum-Optimization/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── setup.py                    # Automated setup script
├── data.yaml                   # YOLO dataset configuration
├── up4000_deploy/              # Deployment scripts
│   ├── deploy_camo_iphone.py   # iPhone Camo (92.8 FPS tested)
│   ├── deploy_script.py        # UP4000/PC webcam
│   └── deploy_tablet_camera.py # Android tablet
├── src/                        # Source code
│   ├── models/train_model.py   # Training script
│   └── data/                   # Data processing
└── results/                    # Training outputs (auto-generated)
```

## Performance Benchmarks

| Device | FPS | Inference Time | Power Usage |
|--------|-----|----------------|-------------|
| RTX 3060 (GPU) | 92.8 | 10.8ms | ~150W |
| Intel i7 (CPU) | 8-12 | 110ms | ~65W |
| Intel UP4000 | 30+ | 30ms | ~10W |

## Dataset Information

| Metric | Value |
|--------|-------|
| Total Images | 6,864 |
| Training Set | 5,193 images |
| Validation Set | 1,671 images |
| Classes | 3 (handraise, write, read) |
| Dataset Size | ~1.4 GB (excluded from Git) |

## Privacy & Security

- On-device processing: All analysis performed locally
- Student privacy: Facial features not stored or transmitted
- GDPR compliant: Only behavioral patterns analyzed
- Local storage: All data remains on edge device

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- Issues: GitHub Issues for bug reports
- Questions: GitHub Discussions
- Documentation: Comprehensive guides in /docs folder

## Acknowledgments

- Ultralytics YOLOv8: Object detection framework
- Intel OpenVINO: Edge AI optimization toolkit
- Intel AI Training Program 2024: Hardware support and guidance

---

**Made for intelligent education through responsible AI**