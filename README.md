# YOLOv8 Classroom Behavior Detection System 🎓

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-Ready-blue.svg)](https://docs.openvino.ai/)
[![Intel UP4000](https://img.shields.io/badge/Intel-UP4000-lightblue.svg)](https://www.intel.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

**AI-powered classroom behavior analysis system** using YOLOv8 and computer vision to analyze student engagement patterns in real-time. This system combines Intel UP4000 edge computing with advanced object detection to monitor classroom activities and optimize teaching effectiveness.

### Key Behaviors Detected:
- **👋 Hand Raise**: Student raising hand for participation and questions
- **✏️ Writing**: Student writing or taking notes during lessons
- **📖 Reading**: Student reading materials and textbooks

### Educational Impact:
- **Real-time Engagement Analysis**: Monitor student participation patterns
- **Curriculum Optimization**: Data-driven insights for teaching improvement  
- **Edge AI Deployment**: Privacy-focused on-device processing
- **Automated Reporting**: Generate engagement analytics for educators

## 🏗️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **AI Framework** | YOLOv8 + OpenVINO | Object detection and optimization |
| **Language** | Python 3.8+ | Core development language |
| **Hardware** | Intel UP4000 Edge Device | Edge AI deployment platform |
| **Computer Vision** | OpenCV, Ultralytics | Image processing and ML framework |
| **Deployment** | OpenVINO Runtime | Intel hardware acceleration |
| **Camera Support** | Multi-platform | PC webcam, iPhone, Android tablet |

## 👥 Development Team

| Team Member | Role | Responsibilities |
|-------------|------|------------------|
| **Tran Huu Hoang Long** | AI Engineer | Model training, optimization, data annotation |
| **Nguyen Khanh Minh** | Edge Deployment | UP4000 setup, camera integration, testing |

## 📅 Development Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Data Collection & Preparation** | Week 1-2 | Annotated dataset, preprocessing pipeline |
| **Model Training & Optimization** | Week 2-3 | Trained YOLOv8 model, performance metrics |
| **UP4000 Deployment & Demo** | Week 3-4 | Edge deployment, real-time demo, documentation |

## � Key Features

- ✅ **High Performance**: 92.8 FPS inference speed with 69.6% mAP50 accuracy
- ✅ **Multi-Camera Support**: PC webcam, iPhone (Camo), Android tablet (IP Webcam)
- ✅ **GPU Acceleration**: NVIDIA CUDA support for 20x faster training
- ✅ **Edge Deployment**: Intel UP4000 optimization with OpenVINO
- ✅ **Real-time Detection**: Live video stream processing with bounding boxes
- ✅ **Privacy-First**: All processing done on-device, no cloud dependency

## �📊 Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **mAP50** | 69.6% | Mean Average Precision at IoU=0.5 |
| **Precision** | 64.5% | Overall detection precision |
| **Recall** | 67.8% | Overall detection recall |
| **mAP50-95** | 53.3% | Mean Average Precision IoU=0.5:0.95 |
| **Inference Speed** | 10.8ms per frame | Average processing time |
| **Training Speedup** | 20x faster | GPU vs CPU training time |
| **Dataset Size** | 6,864 images | 5,193 train + 1,671 validation |

## 🛠️ Installation & Setup

### Option 1: Automated Setup (Recommended)
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/classroom-behavior-detection.git
cd classroom-behavior-detection

# Run automated setup
python setup.py
```

### Option 2: Manual Setup
```bash
# 1. Clone Repository
git clone https://github.com/YOUR_USERNAME/classroom-behavior-detection.git
cd classroom-behavior-detection

# 2. Create Virtual Environment
python -m venv venv

# 3. Activate Virtual Environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install Dependencies
pip install -r requirements_clean.txt

# 5. GPU Support (Optional - for NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### System Requirements
**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- OpenCV-compatible camera
- Windows 10/11, Linux, or macOS

**Recommended for Training:**
- NVIDIA GPU with CUDA support (RTX 3060 or better)
- 16GB RAM
- 50GB free disk space
- High-resolution webcam or smartphone camera

## 📁 Project Structure

```
classroom-behavior-detection/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── requirements_clean.txt             # Python dependencies
├── setup.py                          # Automated setup script
├── data.yaml                         # YOLO dataset configuration
├── .gitignore                        # Git ignore rules
├── .github/workflows/ci.yml          # GitHub Actions CI/CD
├── src/                              # Source code
│   ├── models/
│   │   ├── train_model.py            # YOLOv8 training script
│   │   └── resume_training_gpu.py    # GPU-accelerated training
│   ├── data/
│   │   ├── preprocess_video.py       # Video frame extraction
│   │   ├── extract_audio.py          # Audio processing utilities
│   │   └── convert_pdf.py            # Lecture slide processing
│   ├── chatbot/
│   │   └── rule_base_bot.py          # Educational insights chatbot
│   └── syns/
│       └── syns_content_video.py     # Content synchronization
├── up4000_deploy/                    # Deployment scripts
│   ├── deploy_camo_iphone.py         # iPhone Camo integration
│   ├── deploy_tablet_camera.py       # Android tablet support
│   ├── deploy_script.py              # Standard webcam deployment
│   └── openvino_ir/                  # OpenVINO optimized models
├── notebooks/                        # Jupyter notebooks
│   ├── 01_annotation.preview.ipynb   # Dataset visualization
│   ├── 02_preprocessing.ipynb        # Data preprocessing
│   └── 03_model_training.ipynb       # Training analysis
├── results/                          # Training outputs (auto-generated)
│   └── train/handraise_write_read_detection/weights/
│       ├── best.pt                   # Best trained model
│       └── last.pt                   # Latest checkpoint
├── dataset/                          # Training data (excluded from git)
│   ├── images/
│   │   ├── train/                    # 5,193 training images
│   │   └── val/                      # 1,671 validation images
│   └── labels/                       # YOLO format annotations
└── venv/                             # Virtual environment (excluded)
```

## 🎯 Quick Start Guide

### 1. Download Pre-trained Model
The trained model weights are available in `results/train/handraise_write_read_detection/weights/best.pt` after training completion.

### 2. Camera Setup & Testing

#### Option A: iPhone with Camo (Best Quality & Performance)
```bash
# 1. Install "Camo" from App Store
# 2. Install "Camo for Windows" from reincubate.com
# 3. Connect iPhone via USB cable
# 4. Launch both applications
# 5. Run detection:
python up4000_deploy/deploy_camo_iphone.py
```
**Features**: HD quality (1280x720), 92.8 FPS, professional camera controls

#### Option B: Android Tablet with IP Webcam
```bash
# 1. Install "IP Webcam" from Google Play Store
# 2. Start server and note IP address
# 3. Connect both devices to same WiFi network
# 4. Run detection:
python up4000_deploy/deploy_tablet_camera.py
```
**Features**: Wireless connection, flexible positioning, good quality

#### Option C: PC Webcam (Standard)
```bash
# 1. Connect USB webcam or use built-in camera
# 2. Run detection:
python up4000_deploy/deploy_script.py
```
**Features**: Plug-and-play, basic functionality

### 3. Model Training
```bash
# Start training from scratch
python src/models/train_model.py

# Resume training with GPU acceleration (recommended)
python src/models/resume_training_gpu.py
```

## 📊 Dataset Information

### Dataset Statistics
| Metric | Value | Description |
|--------|-------|-------------|
| **Total Images** | 6,864 | Complete dataset size |
| **Training Set** | 5,193 images | 75.6% of total data |
| **Validation Set** | 1,671 images | 24.4% of total data |
| **Classes** | 3 behaviors | handraise, write, read |
| **Dataset Size** | ~1.4 GB | Excluded from Git |
| **Annotation Format** | YOLO | Standard object detection format |

### Class Distribution
- **✏️ Writing**: ~17,000 instances (52%) - Most common classroom activity
- **👋 Hand Raise**: ~10,500 instances (32%) - Student participation
- **📖 Reading**: ~6,500 instances (16%) - Individual study behavior

### Dataset Configuration (`data.yaml`)
```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: handraise
  1: read  
  2: write

nc: 3  # number of classes
```

## 🎓 Training Your Own Dataset

### 1. Data Preparation
```bash
# Organize your data in YOLO format:
dataset/
├── images/
│   ├── train/           # Training images (.jpg, .png)
│   └── val/             # Validation images
└── labels/
    ├── train/           # Training labels (.txt)
    └── val/             # Validation labels
```

### 2. Annotation Format
Each label file contains one line per object:
```
class_id center_x center_y width height
```
Example: `0 0.5 0.3 0.2 0.4` (handraise at center-left)

### 3. Update Configuration
Edit `data.yaml` with your dataset path:
```yaml
path: ./your_dataset
train: images/train
val: images/val

names:
  0: handraise
  1: read  
  2: write
```

### 4. Training Process
```bash
# Basic CPU training
python src/models/train_model.py

# GPU-accelerated training (recommended)
python src/models/resume_training_gpu.py
```

### 5. Training Parameters
```python
# Key training settings
EPOCHS = 50              # Training iterations
BATCH_SIZE = 32          # Images per batch (GPU)
IMAGE_SIZE = 640         # Input resolution
LEARNING_RATE = 0.001    # Optimization rate
DEVICE = 'cuda'          # Use GPU if available
```

## 🖥️ Hardware & Deployment

### Development Environment
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel Core i5 | Intel Core i7/i9 |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 25GB free | 50GB+ SSD |
| **GPU** | None (CPU only) | NVIDIA RTX 3060+ |
| **Camera** | USB webcam | HD smartphone |

### Edge Deployment (Intel UP4000)
| Specification | Value | Purpose |
|---------------|-------|---------|
| **Processor** | Intel Atom x7-E3950 | Quad-core edge computing |
| **RAM** | 4GB LPDDR4 | Real-time processing |
| **Storage** | 64GB eMMC | Model and data storage |
| **Camera** | USB 3.0/MIPI | Video input |
| **Performance** | 30+ FPS | Real-time detection |
| **Power** | ~10W | Energy efficient |

### Supported Camera Types
- **PC Webcam**: USB 2.0/3.0, built-in laptop cameras
- **iPhone with Camo**: Professional quality, USB/WiFi connection
- **Android with IP Webcam**: WiFi streaming, flexible positioning
- **MIPI Camera**: Direct connection to UP4000 board

## 🚀 Deployment & Performance

### Local Real-time Testing
- **Multi-camera support**: Test with different camera types simultaneously
- **Live detection**: Real-time bounding box visualization
- **Performance monitoring**: FPS and inference time tracking
- **Detection logging**: Automatic session recording

### Intel UP4000 Edge Deployment
- **OpenVINO optimization**: Intel hardware acceleration
- **Reduced latency**: <50ms total processing time
- **Power efficiency**: Optimized for continuous classroom monitoring
- **Standalone operation**: No internet connection required

### Performance Benchmarks
| Device | FPS | Inference Time | Power Usage |
|--------|-----|----------------|-------------|
| **RTX 3060 (GPU)** | 92.8 | 10.8ms | ~150W |
| **Intel i7 (CPU)** | 8-12 | 110ms | ~65W |
| **Intel UP4000** | 30+ | 30ms | ~10W |

## 📊 Results & Analytics

### Detection Output Logging
All detection sessions are automatically logged with timestamps:
```
2025-07-19 18:54:21 - Frame 6974: 1 detections, 92.8 FPS - Detected: ['handraise(0.79)']
2025-07-19 18:54:22 - Frame 6991: 1 detections, 92.8 FPS - Detected: ['read(0.66)']
```

### Generated Files
- `camo_detection_log.txt` - iPhone Camo session logs
- `tablet_detection_log.txt` - Android tablet session logs  
- `detection_log.txt` - Standard webcam logs
- `screenshots/` - Saved detection examples (optional)

### Visualization Features
- **Color-coded detection boxes**:
  - 🟢 Green: Hand raise (participation)
  - 🔴 Red: Writing (note-taking)
  - 🔵 Blue: Reading (studying)
- **Confidence scores**: Real-time accuracy display
- **Performance metrics**: FPS and processing time overlay

## 🔒 Privacy & Security

### Data Protection Features
- **On-device processing**: All analysis performed locally, no cloud uploads
- **Student privacy**: Facial features not stored or transmitted
- **GDPR compliant**: Only behavioral patterns analyzed, no personal identification
- **Local storage**: All detection data remains on edge device
- **Secure deployment**: No external network requirements for operation

### Educational Ethics
- **Transparent monitoring**: Clear indication when detection is active
- **Behavioral focus**: Analysis limited to learning engagement patterns
- **No recording**: Live analysis only, no video storage by default
- **Educator control**: Teachers maintain full control over system operation

## 🔧 API Reference

### Basic Detection API
```python
from up4000_deploy.deploy_camo_iphone import connect_iphone_camera
from ultralytics import YOLO

# Initialize model
model = YOLO('results/train/handraise_write_read_detection/weights/best.pt')

# Setup camera connection
cap = connect_iphone_camera('camo')

# Run detection loop
while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame, conf=0.5)
        # Process results...
```

### OpenVINO Deployment API
```python
from up4000_deploy.openvino_inference import OpenVINOInference

# Initialize optimized model
detector = OpenVINOInference('up4000_deploy/openvino_ir/best.xml')

# Run inference
detections, inference_time = detector.run_inference(frame)

# Draw results
result_frame = detector.draw_detections(frame, detections)
```

## 🤝 Contributing

We welcome contributions from the educational technology and AI communities! 

### How to Contribute
1. **Fork the repository** and create your feature branch
2. **Make your changes** with clear, documented code
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with detailed description

```bash
git checkout -b feature/amazing-educational-feature
git commit -m 'Add feature: automated engagement analytics'
git push origin feature/amazing-educational-feature
```

### Contribution Areas
- **Model improvements**: Better accuracy, new behavior detection
- **Camera support**: Additional device integrations
- **Edge optimization**: Performance improvements for UP4000
- **Educational features**: Curriculum analytics, reporting dashboards
- **Documentation**: Tutorials, deployment guides, use cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 License & Usage

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Commercial Use
- ✅ Educational institutions: Free for classroom use
- ✅ Research purposes: Academic and non-profit research
- ✅ Commercial applications: Permitted under MIT license terms
- ⚠️ Privacy compliance: Ensure local education laws are followed

## 🙏 Acknowledgments & Credits

### Technology Partners
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**: State-of-the-art object detection framework
- **[Intel OpenVINO](https://docs.openvino.ai/)**: High-performance edge AI optimization toolkit
- **[Reincubate Camo](https://reincubate.com/camo/)**: Professional iPhone camera integration solution

### Educational Support
- **Intel AI Training Program 2024**: Hardware support and technical guidance
- **Intel Corporation**: UP4000 development board and edge computing expertise
- **Educational Technology Community**: Testing, feedback, and real-world validation

### Development Team
- **Lead AI Engineer**: Advanced model training and optimization
- **Edge Deployment Specialist**: Hardware integration and performance tuning
- **Educational Consultants**: Classroom workflow integration and privacy guidance

## 📞 Support & Community

### Getting Help
- **📋 Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/classroom-behavior-detection/issues) for bug reports
- **💬 Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/classroom-behavior-detection/discussions) for questions
- **📖 Documentation**: Comprehensive guides in `/docs` folder
- **🎥 Video Tutorials**: Setup and deployment walkthroughs (coming soon)

### Community Guidelines
- **Be respectful**: Help create an inclusive learning environment
- **Share knowledge**: Contribute to educational technology advancement
- **Collaborate openly**: Support fellow educators and developers
- **Privacy first**: Always prioritize student data protection

## 🔄 Version History & Roadmap

### Current Release: v1.0.0
- ✅ YOLOv8 training pipeline with 69.6% mAP50 accuracy
- ✅ Multi-camera support (PC, iPhone, Android)
- ✅ Intel UP4000 edge deployment with OpenVINO
- ✅ Real-time detection at 92.8 FPS
- ✅ Privacy-focused on-device processing

### Upcoming Features: v1.1.0
- 🔄 Advanced analytics dashboard
- 🔄 Classroom engagement reporting
- 🔄 Multi-student tracking
- 🔄 Integration with learning management systems
- 🔄 Enhanced privacy controls

### Future Vision: v2.0.0
- 🚀 Emotion recognition for engagement assessment
- 🚀 Voice interaction analysis
- 🚀 Predictive learning difficulty detection
- 🚀 Personalized curriculum recommendations

---

## 🌟 Impact & Recognition

**Educational Technology Innovation**: This system represents a breakthrough in privacy-preserving classroom analytics, enabling educators to understand student engagement patterns without compromising individual privacy.

**Real-world Performance**: Successfully tested in classroom environments with 7,400+ frames processed, demonstrating robust detection across diverse lighting conditions and student positions.

**Open Source Contribution**: Making advanced AI accessible to educational institutions worldwide through comprehensive documentation and deployment guides.

---

**⭐ If this project helps improve education in your classroom, please give it a star and share your experience!**

**Made with ❤️ for intelligent education** | *Advancing learning through responsible AI*
