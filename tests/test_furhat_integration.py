#!/usr/bin/env python3
"""
Test script for Furhat Multimodal Emotion Recognition System
Tests the core functionality without requiring an actual Furhat robot connection.
"""

import sys
import os

# Add project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        print("✅ Transformers available")
    except ImportError:
        print("❌ Transformers not available")
        return False
    
    try:
        from furhat_remote_api import FurhatRemoteAPI
        print("✅ Furhat Remote API available")
    except ImportError:
        print("❌ Furhat Remote API not available")
        return False
    
    try:
        import src.furhat.furhat_multimodal_emotion_inference as fmei
        print("✅ Furhat multimodal script importable")
    except ImportError as e:
        print(f"❌ Furhat multimodal script import failed: {e}")
        return False
    
    return True

def test_model_files():
    """Test if model files exist"""
    print("\n🔍 Testing model files...")
    
    all_found = True
    
    # Check FER model
    fer_model_path = "models/fer2013_final_model.pth"
    if os.path.exists(fer_model_path):
        print(f"✅ FER model found: {fer_model_path}")
    else:
        print(f"⚠️  FER model not found: {fer_model_path}")
        all_found = False
    
    # Check TER model directory
    ter_model_path = "models/ter_distilbert_model"
    if os.path.exists(ter_model_path):
        print(f"✅ TER model directory found: {ter_model_path}")
        
        # Check key files
        key_files = ["config.json", "model.safetensors", "label_encoder.pkl"]
        for file in key_files:
            file_path = os.path.join(ter_model_path, file)
            if os.path.exists(file_path):
                print(f"  ✅ {file}")
            else:
                print(f"  ⚠️  {file} missing")
                all_found = False
    else:
        print(f"⚠️  TER model directory not found: {ter_model_path}")
        all_found = False
    
    return all_found

def test_class_initialization():
    """Test class initialization without robot connection"""
    print("\n🔍 Testing class initialization...")
    
    try:
        # Import the class
        from src.furhat.furhat_multimodal_emotion_inference import FurhatMultimodalEmotionInference
        
        print("✅ Class import successful")
        
        # Test initialization with no robot connection (should handle gracefully)
        print("🔄 Testing initialization (will show connection errors - this is expected)...")
        
        # This will fail to connect to robot but should initialize other components
        system = FurhatMultimodalEmotionInference(
            furhat_ip="localhost",  # Will fail to connect
            enable_robot_responses=False
        )
        
        print("✅ Class initialization completed (robot connection failed as expected)")
        
        # Test emotion mappings
        print(f"✅ Emotion labels: {system.emotion_labels}")
        print(f"✅ Emotion colors defined: {len(system.emotion_colors)} colors")
        
        # Import the constant from the module
        from src.furhat.furhat_multimodal_emotion_inference import EMOTION_GESTURES
        print(f"✅ Emotion gestures defined: {len(EMOTION_GESTURES)} gestures")
        
        return True
        
    except Exception as e:
        print(f"❌ Class initialization test failed: {e}")
        return False

def test_emotion_fusion():
    """Test emotion fusion logic"""
    print("\n🔍 Testing emotion fusion...")
    
    try:
        from src.furhat.furhat_multimodal_emotion_inference import FurhatMultimodalEmotionInference
        
        # Create a minimal instance for testing
        system = FurhatMultimodalEmotionInference.__new__(FurhatMultimodalEmotionInference)
        system.fusion_strategy = 'confidence_based'
        
        # Test fusion methods
        fused_emotion, fused_confidence = system._confidence_based_fusion(
            'happy', 0.8, 'happy', 0.6
        )
        print(f"✅ Agreeing emotions fusion: {fused_emotion} ({fused_confidence:.3f})")
        
        fused_emotion, fused_confidence = system._confidence_based_fusion(
            'happy', 0.8, 'sad', 0.6
        )
        print(f"✅ Disagreeing emotions fusion: {fused_emotion} ({fused_confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Emotion fusion test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 Furhat Multimodal Emotion Recognition - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model File Tests", test_model_files),
        ("Class Initialization", test_class_initialization),
        ("Emotion Fusion", test_emotion_fusion)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The Furhat integration is ready to use.")
        print("\n📋 Next steps:")
        print("1. Ensure your Furhat robot is connected and Remote API is enabled")
        print("2. Run: python furhat_multimodal_emotion_inference.py --furhat_ip <robot_ip>")
        print("3. Press 'V' to activate voice capture and start speaking to the robot")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
        
        if not any(name == "Import Tests" and result for name, result in results):
            print("\n💡 Install missing dependencies with:")
            print("pip install -r requirements_furhat.txt")

if __name__ == "__main__":
    main()
