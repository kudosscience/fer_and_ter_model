#!/usr/bin/env python3
"""
Example usage and demonstration of the FER camera inference system
This script shows how to use the camera inference without requiring an actual camera
"""

import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def show_usage_examples():
    """Show different ways to use the camera inference script"""
    print("📹 FER Camera Inference Usage Examples")
    print("=" * 50)
    
    print("\n1. Basic usage (auto-detect camera and device):")
    print("   python camera_fer_inference.py")
    
    print("\n2. Use specific camera ID:")
    print("   python camera_fer_inference.py --camera_id 1")
    
    print("\n3. Use specific model file:")
    print("   python camera_fer_inference.py --model_path my_custom_model.pth")
    
    print("\n4. Force CPU usage (disable GPU):")
    print("   python camera_fer_inference.py --device cpu")
    
    print("\n5. Force CUDA usage:")
    print("   python camera_fer_inference.py --device cuda")
    
    print("\n6. Combined options:")
    print("   python camera_fer_inference.py --model_path fer2013_final_model.pth --camera_id 0 --device auto")

def show_interactive_controls():
    """Show the interactive controls available during inference"""
    print("\n🎮 Interactive Controls During Inference")
    print("=" * 50)
    
    controls = [
        ("Q", "Quit the application"),
        ("S", "Save current frame with predictions"),
        ("F", "Toggle fullscreen mode"),
        ("ESC", "Alternative quit key")
    ]
    
    for key, description in controls:
        print(f"   {key:<3} - {description}")

def show_model_info():
    """Show information about the model"""
    print("\n🧠 Model Information")
    print("=" * 50)
    
    try:
        from camera_fer_inference import EmotionCNN
        import torch
        
        # Load the model to get info
        device = torch.device('cpu')
        model = EmotionCNN(num_classes=7, dropout_rate=0.5)
        
        # Load checkpoint to show accuracy
        checkpoint = torch.load('fer2013_final_model.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            test_accuracy = checkpoint.get('test_accuracy', 'Unknown')
            best_val_loss = checkpoint.get('best_val_loss', 'Unknown')
            
            print(f"📊 Test Accuracy: {test_accuracy:.2%}" if isinstance(test_accuracy, float) else f"📊 Test Accuracy: {test_accuracy}")
            print(f"📉 Best Validation Loss: {best_val_loss:.4f}" if isinstance(best_val_loss, float) else f"📉 Best Validation Loss: {best_val_loss}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🔢 Total Parameters: {total_params:,}")
        
        # Model architecture summary
        print(f"🏗️ Architecture: CNN with 4 conv layers + 3 FC layers")
        print(f"📥 Input: 48x48 grayscale face images")
        print(f"📤 Output: 7 emotion classes")
        
        # Emotion classes
        emotion_labels = {
            0: 'Angry 😠', 1: 'Disgust 🤢', 2: 'Fear 😨', 3: 'Happy 😊',
            4: 'Sad 😢', 5: 'Surprise 😲', 6: 'Neutral 😐'
        }
        
        print(f"\n🎭 Emotion Classes:")
        for i, emotion in emotion_labels.items():
            print(f"   {i}: {emotion}")
        
    except Exception as e:
        print(f"❌ Could not load model info: {e}")
        print("Make sure fer2013_final_model.pth is in the current directory")

def show_troubleshooting():
    """Show common troubleshooting tips"""
    print("\n🔧 Troubleshooting Tips")
    print("=" * 50)
    
    issues = [
        ("Camera not found", [
            "Check if camera is connected and working",
            "Try different camera IDs (0, 1, 2...)",
            "Close other applications using the camera",
            "Check camera permissions"
        ]),
        ("Low FPS / Poor performance", [
            "Use GPU acceleration with --device cuda",
            "Close other resource-intensive applications",
            "Ensure good lighting for face detection",
            "Keep face clearly visible in frame"
        ]),
        ("Poor emotion recognition", [
            "Ensure face is well-lit and clearly visible",
            "Keep face relatively centered in frame",
            "Avoid extreme facial angles",
            "Model works best with frontal face views"
        ]),
        ("Module import errors", [
            "Install required packages: pip install -r requirements_camera_inference.txt",
            "Check Python version (3.7+ required)",
            "Consider using a virtual environment"
        ])
    ]
    
    for issue, solutions in issues:
        print(f"\n❗ {issue}:")
        for solution in solutions:
            print(f"   • {solution}")

def test_prerequisites():
    """Test if all prerequisites are met"""
    print("\n🔍 Checking Prerequisites")
    print("=" * 50)
    
    # Check Python version
    import sys
    print(f"🐍 Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        ('torch', 'PyTorch for deep learning'),
        ('torchvision', 'PyTorch vision utilities'),
        ('cv2', 'OpenCV for computer vision'),
        ('numpy', 'Numerical computing'),
        ('PIL', 'Python Imaging Library')
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:<12} - {description}")
        except ImportError:
            print(f"❌ {package:<12} - {description} (MISSING)")
            missing_packages.append(package)
    
    # Check model file
    if os.path.exists('fer2013_final_model.pth'):
        print(f"✅ Model file    - fer2013_final_model.pth found")
    else:
        print(f"❌ Model file    - fer2013_final_model.pth (MISSING)")
        missing_packages.append('model_file')
    
    # Check camera (basic check)
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"✅ Camera       - Default camera accessible")
            cap.release()
        else:
            print(f"⚠️ Camera       - Default camera not accessible")
    except:
        print(f"❌ Camera       - Cannot test camera access")
    
    if missing_packages:
        print(f"\n⚠️ Missing requirements:")
        for package in missing_packages:
            if package == 'model_file':
                print(f"   • Run the training notebook to generate fer2013_final_model.pth")
            else:
                print(f"   • pip install {package}")
        return False
    else:
        print(f"\n✅ All prerequisites met! Ready to run camera inference.")
        return True

def main():
    """Main demonstration function"""
    print("🎥 FER Camera Inference - Usage Guide and Examples")
    print("=" * 60)
    
    # Test prerequisites first
    prerequisites_ok = test_prerequisites()
    
    # Show usage examples
    show_usage_examples()
    
    # Show interactive controls
    show_interactive_controls()
    
    # Show model information
    show_model_info()
    
    # Show troubleshooting tips
    show_troubleshooting()
    
    # Final recommendations
    print("\n🚀 Getting Started")
    print("=" * 50)
    
    if prerequisites_ok:
        print("1. Run: python camera_fer_inference.py")
        print("2. Position your face in front of the camera")
        print("3. Watch real-time emotion predictions!")
        print("4. Press 'Q' to quit when done")
    else:
        print("1. Install missing packages first")
        print("2. Ensure model file is present")
        print("3. Then run: python camera_fer_inference.py")
    
    print(f"\n💡 For more details, see README_camera_inference.md")

if __name__ == "__main__":
    main()
