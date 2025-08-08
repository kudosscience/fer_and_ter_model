#!/usr/bin/env python3
"""
Installation and Setup Script for Voice TER Inference

This script helps users set up the environment for voice-based
textual emotion recognition.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a system command with error handling"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please upgrade to Python 3.7 or later")
        return False

def check_system_dependencies():
    """Check and install system dependencies based on platform"""
    system = platform.system().lower()
    
    print(f"📱 Detected system: {system}")
    
    if system == "linux":
        print("🐧 Linux system detected")
        print("📦 Installing system dependencies...")
        
        commands = [
            ("sudo apt-get update", "Updating package lists"),
            ("sudo apt-get install -y portaudio19-dev python3-pyaudio", "Installing PortAudio"),
            ("sudo apt-get install -y espeak espeak-data libespeak1 libespeak-dev", "Installing speech libraries")
        ]
        
        for command, description in commands:
            print(f"Running: {command}")
            response = input(f"Execute this command? (y/n): ").lower()
            if response == 'y':
                run_command(command, description)
            else:
                print(f"⏭️  Skipped: {description}")
    
    elif system == "darwin":  # macOS
        print("🍎 macOS system detected")
        print("📦 Installing system dependencies...")
        
        # Check if Homebrew is installed
        homebrew_check = subprocess.run("which brew", shell=True, capture_output=True)
        if homebrew_check.returncode != 0:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
        
        run_command("brew install portaudio", "Installing PortAudio via Homebrew")
    
    elif system == "windows":
        print("🪟 Windows system detected")
        print("💡 For Windows, we recommend using conda for PyAudio installation")
        print("   conda install pyaudio")
        print("   Or try: pip install pipwin && pipwin install pyaudio")
    
    return True

def install_python_packages():
    """Install Python packages"""
    print("\n📦 Installing Python packages...")
    
    # Check if requirements file exists
    requirements_file = "requirements_voice_ter.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file '{requirements_file}' not found")
        print("Creating basic requirements...")
        
        basic_requirements = """torch>=2.0.0
transformers>=4.21.0
scikit-learn>=1.0.0
numpy>=1.21.0
speechrecognition>=3.8.1
pyaudio>=0.2.11
tqdm>=4.64.0
pandas>=1.5.0"""
        
        with open(requirements_file, 'w') as f:
            f.write(basic_requirements)
        
        print(f"✅ Created {requirements_file}")
    
    # Install packages
    pip_command = f"{sys.executable} -m pip install -r {requirements_file}"
    success = run_command(pip_command, "Installing Python packages")
    
    if not success:
        print("\n💡 Alternative installation methods:")
        print("1. Try updating pip: python -m pip install --upgrade pip")
        print("2. Install packages individually:")
        
        packages = ["torch", "transformers", "scikit-learn", "numpy", "speechrecognition", "tqdm", "pandas"]
        for package in packages:
            print(f"   pip install {package}")
        
        print("3. For PyAudio issues:")
        print("   - Windows: conda install pyaudio")
        print("   - Linux: sudo apt-get install python3-pyaudio")
        print("   - macOS: brew install portaudio && pip install pyaudio")
    
    return success

def test_installation():
    """Test if installation was successful"""
    print("\n🧪 Testing installation...")
    
    # Test imports
    test_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("sklearn", "Scikit-learn"),
        ("numpy", "NumPy"),
        ("speech_recognition", "Speech Recognition"),
        ("pandas", "Pandas")
    ]
    
    all_passed = True
    
    for package, name in test_packages:
        try:
            __import__(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - FAILED")
            all_passed = False
    
    # Test PyAudio separately (common issue)
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        p.terminate()
        print(f"✅ PyAudio - OK ({device_count} audio devices found)")
    except ImportError:
        print("❌ PyAudio - FAILED (package not installed)")
        all_passed = False
    except Exception as e:
        print(f"⚠️  PyAudio - WARNING ({str(e)})")
    
    if all_passed:
        print("\n🎉 Installation test PASSED!")
        print("✅ Ready to use voice TER inference!")
    else:
        print("\n⚠️  Installation test had issues.")
        print("💡 Check error messages above and install missing packages.")
    
    return all_passed

def create_test_script():
    """Create a simple test script"""
    test_script = """#!/usr/bin/env python3
# Quick test script for voice TER inference
import sys
try:
    print("Testing imports...")
    import torch
    import transformers
    import speech_recognition as sr
    import numpy as np
    import pandas as pd
    print("✅ All imports successful!")
    
    # Test device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Computation device: {device}")
    
    # Test speech recognition setup
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("🎤 Microphone access: OK")
    
    print("\\n🎉 System ready for voice TER inference!")
    print("Run: python voice_ter_inference.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing packages")
    sys.exit(1)
except Exception as e:
    print(f"⚠️  Warning: {e}")
    print("System may still work, but check the error above")
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    print("📝 Created test_setup.py")
    print("💡 Run 'python test_setup.py' to test your installation")

def main():
    """Main setup function"""
    print("🎙️  Voice TER Inference - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check system dependencies
    print(f"\n🔧 System Dependencies")
    print("-" * 30)
    check_system_dependencies()
    
    # Install Python packages
    print(f"\n📦 Python Packages")
    print("-" * 30)
    install_success = install_python_packages()
    
    # Test installation
    print(f"\n🧪 Installation Test")
    print("-" * 30)
    test_success = test_installation()
    
    # Create test script
    print(f"\n📝 Additional Setup")
    print("-" * 30)
    create_test_script()
    
    # Final summary
    print(f"\n📊 Setup Summary")
    print("=" * 50)
    
    if install_success and test_success:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Run the demo: python demo_voice_ter.py")
        print("2. Try voice inference: python voice_ter_inference.py")
        print("3. Read the documentation: README_voice_ter.md")
    else:
        print("⚠️  Setup completed with issues.")
        print("\n🔧 Troubleshooting:")
        print("1. Check error messages above")
        print("2. Install missing packages manually")
        print("3. Run: python test_setup.py")
        print("4. See README_voice_ter.md for detailed instructions")
    
    return install_success and test_success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup error: {str(e)}")
        sys.exit(1)
