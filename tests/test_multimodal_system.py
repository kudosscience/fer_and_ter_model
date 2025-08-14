#!/usr/bin/env python3
"""
Test script for Multimodal Emotion Recognition System
Tests individual components and basic functionality
"""

import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image

# Add project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_requirements():
    """Test if all required packages are installed"""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        import torchvision
        import transformers
        import sklearn
        import cv2
        import PIL
        import speech_recognition
        import pyaudio
        print("✅ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def test_device_availability():
    """Test CUDA availability"""
    print("\n🧪 Testing device availability...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, will use CPU")
    
    print(f"PyTorch version: {torch.__version__}")
    return True

def test_camera():
    """Test camera availability"""
    print("\n🧪 Testing camera availability...")
    
    # Test default camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✅ Camera 0 working")
            print(f"   Frame shape: {frame.shape}")
            cap.release()
            return True
        else:
            print("❌ Camera 0 opened but can't read frames")
            cap.release()
            return False
    else:
        print("❌ Camera 0 not available")
        return False

def test_microphone():
    """Test microphone availability"""
    print("\n🧪 Testing microphone availability...")
    
    try:
        import speech_recognition as sr
        
        # List available microphones
        mic_list = sr.Microphone.list_microphone_names()
        print(f"✅ Found {len(mic_list)} microphone(s):")
        for i, mic in enumerate(mic_list[:3]):  # Show first 3
            print(f"   {i}: {mic}")
        
        # Test default microphone
        r = sr.Recognizer()
        m = sr.Microphone()
        
        with m as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
        
        print("✅ Microphone initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False

def test_fer_model():
    """Test FER model loading"""
    print("\n🧪 Testing FER model...")
    
    model_path = "models/fer2013_final_model.pth"
    if not os.path.exists(model_path):
        print(f"⚠️  FER model not found: {model_path}")
        return False
    
    try:
        # Import the model class from our script
        sys.path.append('.')
        from src.multimodal.multimodal_emotion_inference import EmotionCNN
        
        # Load model
        model = EmotionCNN(num_classes=7, dropout_rate=0.5)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Test with dummy input
        dummy_input = torch.randn(1, 1, 48, 48)
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✅ FER model loaded and tested successfully")
        print(f"   Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ FER model test failed: {e}")
        return False

def test_ter_model():
    """Test TER model loading"""
    print("\n🧪 Testing TER model...")
    
    model_dirs = [
        "./models/ter_distilbert_model",
        "./ter_distilbert_model"
    ]
    
    model_dir = None
    for path in model_dirs:
        if os.path.exists(path):
            model_dir = path
            break
    
    if model_dir is None:
        print("⚠️  TER model directory not found, will test with pre-trained model")
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', 
                num_labels=7
            )
            
            # Test with dummy text
            test_text = "I am feeling happy today"
            encoding = tokenizer(test_text, return_tensors='pt', truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**encoding)
            
            print("✅ Pre-trained TER model loaded and tested successfully")
            return True
            
        except Exception as e:
            print(f"❌ Pre-trained TER model test failed: {e}")
            return False
    
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        
        # Test with dummy text
        test_text = "I am feeling happy today"
        encoding = tokenizer(test_text, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**encoding)
        
        print(f"✅ Custom TER model loaded from {model_dir}")
        print(f"   Output shape: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Custom TER model test failed: {e}")
        return False

def test_face_detection():
    """Test face detection"""
    print("\n🧪 Testing face detection...")
    
    try:
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create a test image with a simple face-like pattern
        test_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(test_img, (100, 100), 80, 255, -1)  # Face
        cv2.circle(test_img, (80, 80), 10, 0, -1)       # Left eye
        cv2.circle(test_img, (120, 80), 10, 0, -1)      # Right eye
        cv2.ellipse(test_img, (100, 120), (20, 10), 0, 0, 180, 0, -1)  # Mouth
        
        # Detect faces
        faces = face_cascade.detectMultiScale(test_img, 1.1, 4)
        
        print(f"✅ Face detection initialized")
        print(f"   Detected {len(faces)} face(s) in test image")
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🔬 Multimodal Emotion Recognition - System Test")
    print("=" * 60)
    
    tests = [
        ("Package Requirements", test_requirements),
        ("Device Availability", test_device_availability),
        ("Camera", test_camera),
        ("Microphone", test_microphone),
        ("Face Detection", test_face_detection),
        ("FER Model", test_fer_model),
        ("TER Model", test_ter_model),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🔬 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System ready to run.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        print("\n💡 Common solutions:")
        print("   - Install missing packages: pip install -r requirements_multimodal.txt")
        print("   - Check model files are in the correct locations")
        print("   - Verify camera and microphone permissions")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
