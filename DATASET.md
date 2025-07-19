# Dataset Information

The dataset is **NOT included** in this Git repository due to size limitations (~1.4 GB).

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Images** | 6,864 |
| **Training Set** | 5,193 images |
| **Validation Set** | 1,671 images |
| **Classes** | 3 (handraise, write, read) |
| **Total Size** | ~1.4 GB |
| **Format** | YOLO annotation format |

## 📁 Expected Dataset Structure

```
dataset/
├── processed/
│   ├── images/
│   │   ├── train/               # 5,193 training images
│   │   │   ├── image001.jpg
│   │   │   ├── image002.jpg
│   │   │   └── ...
│   │   └── val/                 # 1,671 validation images
│   │       ├── val_image001.jpg
│   │       ├── val_image002.jpg
│   │       └── ...
│   └── labels/
│       ├── train/               # Training labels (YOLO format)
│       │   ├── image001.txt
│       │   ├── image002.txt
│       │   └── ...
│       └── val/                 # Validation labels (YOLO format)
│           ├── val_image001.txt
│           ├── val_image002.txt
│           └── ...
├── raw/                         # Original videos and images
├── annotations/                 # Raw annotation files
├── slides/                      # Lecture slides (PDF, PPTX)
└── transcripts/                 # Speech-to-text transcriptions
```

## 🎯 Class Labels

```yaml
names:
  0: handraise  # Student raising hand for participation
  1: write      # Student writing or taking notes
  2: read       # Student reading materials
```

## 📈 Class Distribution

- **write**: ~17,000 instances (52%) - Most common behavior
- **handraise**: ~10,500 instances (32%) - Moderate frequency  
- **read**: ~6,500 instances (16%) - Least common behavior

## 🔧 Label Format

Each label file (`.txt`) contains bounding box annotations in YOLO format:
```
class_id x_center y_center width height
```

Where all values are normalized between 0 and 1.

**Example label file content:**
```
0 0.5 0.3 0.4 0.6
1 0.2 0.7 0.3 0.2
```

## 🚀 Setup Instructions

### 1. Contact for Dataset Access
```bash
# Contact project maintainers for dataset download link:
# - Tran Huu Hoang Long (AI Engineer)
# - Nguyen Khanh Minh (Hardware Lead)
```

### 2. Download and Extract
```bash
# Extract dataset to project root
# Ensure the folder structure matches above
```

### 3. Verify Dataset
```python
# Run this Python script to verify dataset structure
import os

def verify_dataset():
    base_path = 'dataset/processed'
    
    # Check required directories
    required_dirs = [
        'images/train', 'images/val',
        'labels/train', 'labels/val'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if os.path.exists(full_path):
            file_count = len([f for f in os.listdir(full_path) 
                            if not f.startswith('.')])
            print(f"✅ {dir_path}: {file_count} files")
        else:
            print(f"❌ Missing: {dir_path}")
    
    # Check expected counts
    train_images = len([f for f in os.listdir(f'{base_path}/images/train') 
                       if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_images = len([f for f in os.listdir(f'{base_path}/images/val') 
                     if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Training images: {train_images}")
    print(f"   Validation images: {val_images}")
    print(f"   Total: {train_images + val_images}")
    
    if train_images == 5193 and val_images == 1671:
        print("✅ Dataset verification passed!")
    else:
        print("⚠️ Dataset counts don't match expected values")

# Run verification
if __name__ == "__main__":
    verify_dataset()
```

### 4. Update data.yaml
Ensure your `data.yaml` points to the correct dataset paths:
```yaml
train: dataset/processed/images/train
val: dataset/processed/images/val
nc: 3

names:
  0: handraise
  1: write
  2: read
```

## 🔒 Privacy and Ethics

- **Student Privacy**: All faces should be blurred or anonymized
- **Data Security**: Dataset contains sensitive classroom recordings
- **Usage Rights**: Dataset is for educational/research purposes only
- **GDPR Compliance**: Ensure proper consent and data handling

## 📞 Support

For dataset access, issues, or questions:
- Create an issue in this repository
- Contact the project team directly
- Refer to the main README.md for contact information

---

**Note**: This dataset is specifically created for Intel AI Training Program and classroom behavior analysis research.
