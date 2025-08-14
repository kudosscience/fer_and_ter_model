#!/usr/bin/env python3
"""
Demo script for Furhat Multimodal Emotion Recognition System
Shows how to use the system with different configurations.
"""

import sys
import time
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header():
    """Print demo header"""
    print("🤖 Furhat Multimodal Emotion Recognition Demo")
    print("=" * 50)
    print("This demo shows how to use the Furhat integration")
    print("=" * 50)

def demo_basic_usage():
    """Demo basic usage without robot connection"""
    print("\n📋 Demo 1: Basic Usage (No Robot Required)")
    print("-" * 40)
    
    try:
        from src.furhat.furhat_multimodal_emotion_inference import FurhatMultimodalEmotionInference
        
        print("Creating system instance...")
        system = FurhatMultimodalEmotionInference(
            furhat_ip="localhost",
            enable_robot_responses=False  # Disable responses for demo
        )
        
        print("✅ System created successfully!")
        print(f"Supported emotions: {', '.join(system.emotion_labels)}")
        
        # Demo emotion fusion
        print("\n🔄 Testing emotion fusion...")
        
        test_cases = [
            ("happy", 0.8, "happy", 0.6),
            ("sad", 0.7, "angry", 0.5),
            ("neutral", 0.4, None, None),
            (None, None, "surprise", 0.9)
        ]
        
        for fer_emotion, fer_conf, ter_emotion, ter_conf in test_cases:
            fused_emotion, fused_conf = system._fuse_emotions(
                fer_emotion, fer_conf, ter_emotion, ter_conf
            )
            print(f"  FER: {fer_emotion or 'None'}({fer_conf or 0:.2f}) + "
                  f"TER: {ter_emotion or 'None'}({ter_conf or 0:.2f}) "
                  f"→ {fused_emotion}({fused_conf:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_text_emotion_prediction():
    """Demo text emotion prediction"""
    print("\n📋 Demo 2: Text Emotion Recognition")
    print("-" * 40)
    
    try:
        from src.furhat.furhat_multimodal_emotion_inference import FurhatMultimodalEmotionInference
        
        # Create system (will show connection warnings - that's OK)
        system = FurhatMultimodalEmotionInference(
            furhat_ip="localhost",
            enable_robot_responses=False
        )
        
        if system.ter_model is None:
            print("⚠️  TER model not available, skipping text demo")
            return True
        
        # Test sentences
        test_sentences = [
            "I am so happy today!",
            "This makes me really angry.",
            "I'm feeling quite sad about this.",
            "What a wonderful surprise!",
            "This is disgusting.",
            "I'm scared about what might happen.",
            "The weather is normal today."
        ]
        
        print("🔄 Testing text emotion recognition...")
        for sentence in test_sentences:
            emotion, confidence = system._predict_ter_emotion(sentence)
            if emotion:
                emoji = system.emotion_emojis.get(emotion, '❓')
                print(f"  '{sentence}' → {emotion} {emoji} ({confidence:.2%})")
            else:
                print(f"  '{sentence}' → No prediction")
        
        return True
        
    except Exception as e:
        print(f"❌ Text demo failed: {e}")
        return False

def demo_robot_commands():
    """Demo robot command mapping"""
    print("\n📋 Demo 3: Robot Command Mapping")
    print("-" * 40)
    
    try:
        from src.furhat.furhat_multimodal_emotion_inference import EMOTION_GESTURES, EMOTION_RESPONSES
        
        print("🤖 Emotion to Gesture Mapping:")
        for emotion, gesture in EMOTION_GESTURES.items():
            print(f"  {emotion.title()}: {gesture or 'No gesture'}")
        
        print("\n💬 Sample Robot Responses:")
        for emotion, responses in EMOTION_RESPONSES.items():
            print(f"  {emotion.title()}: '{responses[0]}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Robot command demo failed: {e}")
        return False

def demo_usage_examples():
    """Show usage examples"""
    print("\n📋 Demo 4: Usage Examples")
    print("-" * 40)
    
    examples = [
        ("Basic usage with local robot", 
         "python src/furhat/furhat_multimodal_emotion_inference.py"),
        
        ("Connect to remote robot", 
         "python src/furhat/furhat_multimodal_emotion_inference.py --furhat_ip 192.168.1.100"),
        
        ("Use custom models", 
         "python src/furhat/furhat_multimodal_emotion_inference.py --fer_model ./models/fer2013_final_model.pth --ter_model ./models/ter_distilbert_model"),
        
        ("Detection only (no robot responses)", 
         "python src/furhat/furhat_multimodal_emotion_inference.py --no_robot_responses"),
        
        ("Use weighted fusion strategy", 
         "python src/furhat/furhat_multimodal_emotion_inference.py --fusion weighted_average"),
        
        ("Force CPU usage", 
         "python src/furhat/furhat_multimodal_emotion_inference.py --device cpu")
    ]
    
    print("💡 Command Line Examples:")
    for i, (description, command) in enumerate(examples, 1):
        print(f"\n{i}. {description}:")
        print(f"   {command}")
    
    print("\n⌨️  Interactive Controls (when running):")
    controls = [
        ("Q", "Quit the application"),
        ("V", "Toggle Furhat voice capture"),
        ("S", "Save current frame"),
        ("T", "Toggle TER panel"),
        ("H", "Show prediction history"),
        ("P", "Show statistics"),
        ("R", "Toggle robot responses"),
        ("G", "Test robot gesture")
    ]
    
    for key, description in controls:
        print(f"   {key}: {description}")
    
    return True

def main():
    """Run all demos"""
    print_header()
    
    demos = [
        ("Basic System", demo_basic_usage),
        ("Text Emotion Recognition", demo_text_emotion_prediction),
        ("Robot Commands", demo_robot_commands),
        ("Usage Examples", demo_usage_examples)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            result = demo_func()
            results.append((demo_name, result))
        except Exception as e:
            print(f"❌ {demo_name} failed: {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)
    
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{status} - {demo_name}")
    
    print(f"\n🎯 Ready to use Furhat Multimodal Emotion Recognition!")
    print("\n📋 Quick Start:")
    print("1. Ensure Furhat robot is connected and Remote API is enabled")
    print("2. Run: python src/furhat/furhat_multimodal_emotion_inference.py --furhat_ip <robot_ip>")
    print("3. Press 'V' to activate voice capture")
    print("4. Speak to the robot and watch it respond emotionally!")
    
    print(f"\n📚 For more information, see README_furhat.md")

if __name__ == "__main__":
    main()
