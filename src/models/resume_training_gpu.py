#!/usr/bin/env python3
"""
Resume YOLOv8 training from checkpoint with GPU acceleration
This script will resume your current training progress and switch to GPU
"""

import os
import sys
from ultralytics import YOLO

def resume_training():
    """Resume training from the last checkpoint with GPU"""
    
    # Path to your current checkpoint
    checkpoint_path = 'results/train/handraise_write_read_detection/weights/last.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        print("Please make sure training has started and checkpoints are saved.")
        return False
    
    print(f"📂 Found checkpoint: {checkpoint_path}")
    
    # Check if GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print("🚀 Will resume training on GPU - much faster!")
            batch_size = 32  # Optimized for RTX 3060
        else:
            device = 'cpu'
            print("⚠️ GPU not available, continuing on CPU")
            batch_size = 8
    except Exception as e:
        device = 'cpu'
        batch_size = 8
        print(f"⚠️ GPU check failed: {e}")
    
    try:
        # Load model from checkpoint
        print("🔄 Loading model from checkpoint...")
        model = YOLO(checkpoint_path)
        
        # Resume training with new device
        print(f"▶️ Resuming training on {device.upper()}...")
        results = model.train(
            data='data.yaml',                    # Path to dataset config  
            epochs=50,                           # Total epochs (will continue from where it left off)
            imgsz=640,                          # Image size
            batch=batch_size,                   # Optimized batch size
            device=device,                      # GPU or CPU
            project='results/train',            # Same project directory
            name='handraise_write_read_detection',  # Same experiment name
            save_period=10,                     # Save checkpoint every 10 epochs
            verbose=True,                       # Verbose output
            amp=True if device == 'cuda' else False,  # Mixed precision for GPU
            cache=True,                         # Cache images for faster training
            workers=8 if device == 'cuda' else 4,     # More workers for GPU
            resume=True                         # Important: Resume from checkpoint
        )
        
        print("✅ Training resumed successfully!")
        print(f"📊 Final results: {results}")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 YOLOv8 Training Resumption with GPU Acceleration")
    print("=" * 60)
    
    # Stop current training first (manual instruction)
    print("\n⚠️ IMPORTANT STEPS:")
    print("1. First, stop your current training (Ctrl+C in the training terminal)")
    print("2. Install PyTorch with CUDA support if not already done")
    print("3. Then run this script to resume with GPU acceleration")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    if resume_training():
        print("\n🎉 Training completed successfully!")
        print("📁 Best model saved at: results/train/handraise_write_read_detection/weights/best.pt")
    else:
        print("\n❌ Training resumption failed!")
        sys.exit(1)
