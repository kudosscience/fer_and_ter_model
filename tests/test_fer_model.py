#!/usr/bin/env python3
"""
Test script for FER model loading and basic functionality
This script tests the model without requiring a camera
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

# Import the model class from the main script
from camera_fer_inference import EmotionCNN


def test_model_loading(model_path='fer2013_final_model.pth'):
    """Test model loading functionality"""
    print("🧪 Testing FER Model Loading...")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model instance
        model = EmotionCNN(num_classes=7, dropout_rate=0.5)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint contains additional info (optimizer, metrics, etc.)
            state_dict = checkpoint['model_state_dict']
            print(f"📊 Test accuracy from checkpoint: {checkpoint.get('test_accuracy', 'N/A')}")
            print(f"📊 Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        else:
            # Checkpoint is just the state dict
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        print("✅ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,}")
        
        return True, model, device
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_inference(model, device):
    """Test inference with dummy data"""
    print("\n🧪 Testing Model Inference...")
    
    try:
        # Create dummy input (batch_size=1, channels=1, height=48, width=48)
        dummy_input = torch.randn(1, 1, 48, 48).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(dummy_input)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence_scores = probabilities.cpu().numpy()[0]
        
        # Emotion labels
        emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
        print("✅ Inference successful")
        print(f"📊 Output shape: {outputs.shape}")
        print(f"🎯 Predicted emotion: {emotion_labels[predicted_class]}")
        print(f"📈 Confidence: {confidence_scores[predicted_class]:.2%}")
        
        # Show all class probabilities
        print("\n📊 All emotion probabilities:")
        for i, (emotion, prob) in enumerate(zip(emotion_labels.values(), confidence_scores)):
            print(f"   {emotion}: {prob:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


def test_preprocessing():
    """Test image preprocessing pipeline"""
    print("\n🧪 Testing Image Preprocessing...")
    
    try:
        # Create a dummy grayscale image (48x48)
        rng = np.random.default_rng(42)  # Use modern numpy random generator
        dummy_image = rng.integers(0, 255, (48, 48), dtype=np.uint8)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(dummy_image)
        
        # Define transforms (same as in main script)
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Apply transforms
        tensor_img = transform(pil_img)
        
        # Add batch dimension
        tensor_img = tensor_img.unsqueeze(0)
        
        print("✅ Preprocessing successful")
        print(f"📊 Input image shape: {dummy_image.shape}")
        print(f"📊 Processed tensor shape: {tensor_img.shape}")
        print(f"📈 Tensor range: [{tensor_img.min():.3f}, {tensor_img.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 FER Model Test Suite")
    print("=" * 50)
    
    # Test 1: Model loading
    result = test_model_loading()
    if not result:
        print("\n❌ Model loading test failed. Cannot proceed with other tests.")
        return
    
    _, model, device = result
    
    # Test 2: Preprocessing
    test_preprocessing()
    
    # Test 3: Inference
    test_inference(model, device)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n💡 If all tests passed, you can run the camera inference script:")
    print("   python camera_fer_inference.py")
    print("\n🎛️ Available options:")
    print("   --model_path PATH     # Specify model file path")
    print("   --camera_id ID        # Specify camera ID (default: 0)")
    print("   --device DEVICE       # Specify device (auto/cpu/cuda)")


if __name__ == "__main__":
    main()
