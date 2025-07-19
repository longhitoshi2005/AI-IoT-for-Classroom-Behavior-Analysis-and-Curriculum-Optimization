#!/usr/bin/env python3
"""
Setup script for YOLOv8 Classroom Behavior Detection System
Automated installation and environment setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run shell command with error handling"""
    print(f"\n📋 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu_support():
    """Check for NVIDIA GPU support"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            return True
    except:
        pass
    print("ℹ️  No NVIDIA GPU detected (CPU-only mode)")
    return False

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("ℹ️  Virtual environment already exists")
        return True
    
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    return True

def install_dependencies(gpu_support=False):
    """Install required packages"""
    
    # Determine activation command based on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install CPU version first
    if not run_command(f"{pip_cmd} install -r requirements_clean.txt", 
                      "Installing core dependencies"):
        return False
    
    # Install GPU version if supported
    if gpu_support:
        print("\n🔥 Installing GPU-accelerated PyTorch...")
        gpu_cmd = (f"{pip_cmd} install torch torchvision torchaudio "
                  "--index-url https://download.pytorch.org/whl/cu121")
        if not run_command(gpu_cmd, "Installing GPU PyTorch"):
            print("⚠️  GPU installation failed, using CPU version")
    
    return True

def download_base_models():
    """Download YOLOv8 base models if not present"""
    models = ["yolov8n.pt", "yolov8s.pt"]
    
    for model in models:
        if not Path(model).exists():
            print(f"📥 Downloading {model}...")
            # The model will be downloaded automatically by ultralytics on first use
    
    return True

def verify_installation():
    """Verify the installation by importing key packages"""
    test_imports = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLOv8/Ultralytics"),
        ("numpy", "NumPy"),
    ]
    
    print("\n🔍 Verifying installation...")
    
    # Determine python command
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    for module, name in test_imports:
        cmd = f'{python_cmd} -c "import {module}; print(f\\"✅ {name}: {{module.__version__ if hasattr(module, \'__version__\') else \'OK\'}}\\""'
        if not run_command(cmd, f"Testing {name}"):
            return False
    
    return True

def main():
    """Main setup function"""
    print("🎓 YOLOv8 Classroom Behavior Detection Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check GPU support
    gpu_support = check_gpu_support()
    
    # Setup virtual environment
    if not setup_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies(gpu_support):
        return False
    
    # Download base models
    if not download_base_models():
        return False
    
    # Verify installation
    if not verify_installation():
        return False
    
    # Success message
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Activate virtual environment:")
    
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Test camera detection:")
    print("   # For iPhone with Camo")
    print("   python up4000_deploy/deploy_camo_iphone.py")
    print("   # For PC webcam")
    print("   python up4000_deploy/deploy_script.py")
    
    print("\n3. Train custom model:")
    print("   python src/models/train_model.py")
    
    print("\n📖 Check README.md for detailed instructions!")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        print("\n❌ Setup failed! Check error messages above.")
        sys.exit(1)
