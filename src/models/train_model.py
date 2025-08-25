import os
import subprocess
import sys
from ultralytics import YOLO

def install_dependencies():
    """Install YOLOv8 (ultralytics) and OpenVINO"""
    print("Installing YOLOv8 and dependencies...")
    
    # Install PyTorch with CUDA support for RTX 3060 (CUDA 12.x compatible)
    print("Installing PyTorch with CUDA support...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu121'])
    
    # Install YOLOv8
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
    
    # Install OpenVINO
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'openvino', 'openvino-dev'])
    
    # Install other requirements
    requirements = [
        'opencv-python>=4.1.1',
        'pillow>=7.1.2',
        'pyyaml>=5.3.1',
        'requests>=2.23.0',
        'scipy>=1.4.1',
        'tqdm>=4.41.0',
        'matplotlib>=3.2.2',
        'seaborn>=0.11.0',
        'pandas>=1.1.4',
        'numpy>=1.18.5'
    ]
    
    for req in requirements:
        subprocess.run([sys.executable, '-m', 'pip', 'install', req])
    
    print("Dependencies installed successfully!")

def check_gpu_availability():
    """Check if GPU is available and return the best device"""
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            print(f" GPU detected: {gpu_name}")
            print(f" GPU memory: {gpu_memory:.1f} GB")
            print(f" Using GPU for training - this will be much faster!")
            return device
        else:
            print("  GPU not detected or PyTorch not installed with CUDA support")
            print(" Using CPU for training - this will be slower")
            return 'cpu'
    except ImportError:
        print("  PyTorch not installed - will use CPU")
        return 'cpu'

def check_dataset_for_3_classes():
    """Check if dataset exists for handraise, write, read"""
    required_dirs = [
        'dataset/processed/images/train',
        'dataset/processed/images/val',
        'dataset/processed/labels/train',
        'dataset/processed/labels/val'
    ]
    
    print("Checking dataset for 3 classes (handraise, write, read)...")
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Missing directory: {dir_path}")
            return False
        
        files = os.listdir(dir_path)
        if not files:
            print(f"Empty directory: {dir_path}")
            return False

        print(f"Found {len(files)} files in {dir_path}")

    # Check if labels have correct class IDs (0, 1, 2)
    print("Validating class IDs in labels...")
    label_files = os.listdir('dataset/processed/labels/train')[:5]

    for label_file in label_files:
        label_path = os.path.join('dataset/processed/labels/train', label_file)
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id not in [0, 1, 2]:
                        print(f"Invalid class_id {class_id} in {label_file}")
                        print("   Valid class IDs: 0=handraise, 1=write, 2=read")
                        return False
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            return False
    
    print("Dataset validation passed for 3 classes!")
    return True

def train_yolo_model(resume_from_checkpoint=False):
    """Train YOLOv8 model"""
    # Check dataset first
    if not check_dataset_for_3_classes():
        print("Dataset validation failed. Please check your dataset structure.")
        return False
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU availability
    device = check_gpu_availability()
    
    # Optimize batch size based on device
    if device == 'cuda':
        batch_size = 32  # RTX 3060 can handle larger batches
        print(" Using optimized settings for GPU training")
    else:
        batch_size = 8   # Conservative for CPU
        print("  Using conservative settings for CPU training")
    
    try:
        # Check for existing checkpoint to resume from
        checkpoint_path = 'results/train/handraise_write_read_detection/weights/last.pt'
        
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            print(f" Found existing checkpoint: {checkpoint_path}")
            print(" Resuming training from checkpoint...")
            model = YOLO(checkpoint_path)  # Load from checkpoint
            print(f" Successfully loaded checkpoint - continuing training on {device.upper()}")
        else:
            # Initialize YOLOv8 model from scratch
            print(" Starting fresh training...")
            model = YOLO('yolov8s.pt')  # Start with YOLOv8 small pretrained model
        
        # Train the model
        print("Starting YOLOv8 training...")
        results = model.train(
            data='data.yaml',                    # Path to dataset config
            epochs=50,                           # Number of epochs
            imgsz=640,                          # Image size
            batch=batch_size,                   # Dynamic batch size based on device
            device=device,                      # Auto-detected device (cuda/cpu)
            project='results/train',            # Project directory
            name='handraise_write_read_detection',  # Experiment name
            save_period=10,                     # Save checkpoint every 10 epochs
            verbose=True,                       # Verbose output
            amp=True if device == 'cuda' else False,  # Automatic Mixed Precision for GPU
            cache=True,                         # Cache images for faster training
            workers=8 if device == 'cuda' else 4  # More workers for GPU
        )
        
        print("YOLOv8 training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def convert_to_openvino():
    """Convert trained YOLOv8 model to OpenVINO IR format"""
    model_path = 'results/train/handraise_write_read_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Please check if training completed successfully.")
        return False

    print(f"Found trained model at: {model_path}")
    
    try:
        # Load the trained model
        model = YOLO(model_path)
        
        # Export to OpenVINO
        print("Converting YOLOv8 to OpenVINO format...")
        model.export(
            format='openvino',                  # Export format
            imgsz=640,                         # Image size
            optimize=True,                     # Optimize for inference
            half=False,                        # Use FP32 (change to True for FP16)
            int8=False,                        # Use INT8 quantization
            dynamic=False,                     # Dynamic input shapes
            simplify=True                      # Simplify model
        )
        
        # Move exported model to deployment directory
        import shutil
        source_dir = model_path.replace('.pt', '_openvino_model')
        dest_dir = 'up4000_deploy/openvino_ir'
        
        os.makedirs(dest_dir, exist_ok=True)
        
        if os.path.exists(source_dir):
            # Copy files
            for file in os.listdir(source_dir):
                src_file = os.path.join(source_dir, file)
                dst_file = os.path.join(dest_dir, file)
                shutil.copy2(src_file, dst_file)
            
            print(f"OpenVINO IR model saved to {dest_dir}")
            return True
        else:
            print("OpenVINO export failed - output directory not found")
            return False
            
    except Exception as e:
        print(f"OpenVINO conversion failed: {e}")
        return False

def resume_training_with_gpu():
    """Resume training from checkpoint with GPU acceleration"""
    print(" Resuming training with GPU acceleration...")
    return train_yolo_model(resume_from_checkpoint=True)

if __name__ == "__main__":
    # Ask user if they want to resume from checkpoint
    resume_choice = input("Do you want to resume from existing checkpoint? (y/n): ").lower().strip()
    
    if resume_choice == 'y' or resume_choice == 'yes':
        # Resume training
        if resume_training_with_gpu():
            print("Training completed successfully!")
            
            # Convert to OpenVINO
            if convert_to_openvino():
                print("OpenVINO conversion completed!")
            else:
                print("OpenVINO conversion failed!")
        else:
            print("Training failed!")
    else:
        # Train from scratch
        if train_yolo_model():
            print("Training completed successfully!")
            
            # Convert to OpenVINO
            if convert_to_openvino():
                print("OpenVINO conversion completed!")
            else:
                print("OpenVINO conversion failed!")
        else:
            print("Training failed!")
