#!/usr/bin/env python3
"""
Demo script for Multimodal Emotion Recognition System
Shows basic usage and example scenarios
"""

import sys
import time
from datetime import datetime

def demo_basic_usage():
    """Demonstrate basic usage of the multimodal system"""
    print("🎭 Multimodal Emotion Recognition Demo")
    print("=" * 50)
    print()
    
    print("📖 This demo shows how to use the multimodal emotion recognition system.")
    print()
    
    print("🚀 Basic Usage:")
    print("   python multimodal_emotion_inference.py")
    print()
    
    print("⚙️  With Custom Options:")
    print("   python multimodal_emotion_inference.py \\")
    print("       --fer_model ./my_fer_model.pth \\")
    print("       --ter_model ./my_ter_model \\")
    print("       --device cuda \\")
    print("       --fusion confidence_based")
    print()

def demo_interactive_controls():
    """Show interactive controls"""
    print("🎮 Interactive Controls")
    print("-" * 30)
    print()
    
    controls = [
        ("V", "Toggle voice capture ON/OFF", "Start/stop listening to microphone"),
        ("Q", "Quit application", "Exit the program safely"),
        ("S", "Save current frame", "Save annotated video frame"),
        ("T", "Toggle TER panel", "Show/hide text emotion panel"),
        ("F", "Toggle fullscreen", "Switch between windowed and fullscreen"),
        ("H", "Show history", "Display recent emotion predictions"),
        ("P", "Print statistics", "Show session emotion statistics")
    ]
    
    for key, action, description in controls:
        print(f"   [{key}] {action}")
        print(f"       {description}")
        print()

def demo_fusion_strategies():
    """Explain fusion strategies"""
    print("🔗 Emotion Fusion Strategies")
    print("-" * 35)
    print()
    
    print("1. 📊 Confidence-Based Fusion (Default)")
    print("   • Selects emotion with highest confidence")
    print("   • Boosts confidence when both modalities agree")
    print("   • Reduces confidence when they disagree")
    print("   • Best for: Balanced scenarios")
    print()
    
    print("2. ⚖️  Weighted Average Fusion")
    print("   • Combines using fixed weights (60% facial, 40% textual)")
    print("   • Consistent weighting regardless of confidence")
    print("   • Best for: When one modality is consistently more reliable")
    print()
    
    print("💡 Usage:")
    print("   python multimodal_emotion_inference.py --fusion confidence_based")
    print("   python multimodal_emotion_inference.py --fusion weighted_average")
    print()

def demo_example_scenarios():
    """Show example usage scenarios"""
    print("🎯 Example Usage Scenarios")
    print("-" * 35)
    print()
    
    scenarios = [
        {
            "title": "🏠 Home Monitoring",
            "description": "Monitor family emotions during video calls",
            "command": "python multimodal_emotion_inference.py --camera_id 0",
            "notes": "Use default camera and models"
        },
        {
            "title": "🏢 Office Environment", 
            "description": "Analyze emotions during meetings",
            "command": "python multimodal_emotion_inference.py --fusion weighted_average --device cuda",
            "notes": "Use GPU acceleration and weighted fusion"
        },
        {
            "title": "🎓 Research Study",
            "description": "Collect emotion data for analysis",
            "command": "python multimodal_emotion_inference.py --ter_model ./custom_model",
            "notes": "Use custom trained models"
        },
        {
            "title": "💻 Low-Power Device",
            "description": "Run on CPU-only systems",
            "command": "python multimodal_emotion_inference.py --device cpu",
            "notes": "Force CPU usage for compatibility"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['title']}")
        print(f"   {scenario['description']}")
        print(f"   Command: {scenario['command']}")
        print(f"   Notes: {scenario['notes']}")
        print()

def demo_understanding_output():
    """Explain the output and UI elements"""
    print("📺 Understanding the Output")
    print("-" * 35)
    print()
    
    print("🖼️  Video Display:")
    print("   • Live camera feed with face detection boxes")
    print("   • Emotion labels and confidence scores on faces")
    print("   • Color-coded emotions (red=angry, yellow=happy, etc.)")
    print()
    
    print("📊 Information Panels:")
    print("   • Top-left: FPS, device info, current emotions")
    print("   • Bottom-left: Voice capture status and recent text")
    print("   • Bottom-right: Fused emotion result")
    print("   • Top-right: Keyboard controls reference")
    print()
    
    print("🎤 Voice Processing:")
    print("   • 'LISTENING' when voice capture is active")
    print("   • Shows transcribed text and TER emotion")
    print("   • Displays confidence scores for text predictions")
    print()
    
    print("🔄 Fusion Results:")
    print("   • Combined emotion from both face and voice")
    print("   • Overall confidence score")
    print("   • Real-time updates as new data arrives")
    print()

def demo_troubleshooting():
    """Common troubleshooting tips"""
    print("🔧 Common Issues & Solutions")
    print("-" * 35)
    print()
    
    issues = [
        {
            "problem": "Camera not working",
            "solutions": [
                "Check camera permissions in system settings",
                "Try different camera ID: --camera_id 1",
                "Ensure no other app is using the camera",
                "Test with: python test_multimodal_system.py"
            ]
        },
        {
            "problem": "Microphone not responding",
            "solutions": [
                "Check microphone permissions",
                "Verify microphone is set as default device",
                "Test audio levels in system settings",
                "Try pressing 'V' to toggle voice capture"
            ]
        },
        {
            "problem": "Low FPS / Performance issues",
            "solutions": [
                "Use CUDA if available: --device cuda",
                "Close other applications using camera/mic",
                "Reduce camera resolution in code",
                "Use CPU if GPU is overloaded: --device cpu"
            ]
        },
        {
            "problem": "Model not found errors",
            "solutions": [
                "Ensure fer2013_final_model.pth is in directory",
                "Check ter_distilbert_model/ folder exists",
                "Use absolute paths to model files",
                "Run test_multimodal_system.py to verify"
            ]
        },
        {
            "problem": "Voice recognition not working",
            "solutions": [
                "Check internet connection (Google Speech API)",
                "Speak clearly and close to microphone",
                "Reduce background noise",
                "Wait for 'LISTENING' status before speaking"
            ]
        }
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. ❌ {issue['problem']}")
        for solution in issue['solutions']:
            print(f"   ✅ {solution}")
        print()

def demo_system_requirements():
    """Show system requirements"""
    print("💻 System Requirements")
    print("-" * 25)
    print()
    
    print("🔧 Hardware:")
    print("   • Camera/webcam (USB or built-in)")
    print("   • Microphone (USB, built-in, or headset)")
    print("   • RAM: 4GB minimum, 8GB recommended")
    print("   • Storage: 2GB for models and dependencies")
    print("   • GPU: Optional but recommended for better performance")
    print()
    
    print("💿 Software:")
    print("   • Python 3.8 or higher")
    print("   • Windows 10/11, macOS 10.14+, or Linux")
    print("   • Internet connection (for voice recognition)")
    print("   • Camera/microphone drivers and permissions")
    print()
    
    print("📦 Python Packages:")
    print("   • PyTorch + torchvision")
    print("   • OpenCV (cv2)")
    print("   • Transformers (Hugging Face)")
    print("   • SpeechRecognition + PyAudio")
    print("   • scikit-learn, numpy, PIL")
    print()
    
    print("💡 Install with:")
    print("   pip install -r requirements_multimodal.txt")
    print()

def main():
    """Run the demo"""
    print("🎭 MULTIMODAL EMOTION RECOGNITION")
    print("🎯 SYSTEM DEMO & GUIDE")
    print("=" * 60)
    print()
    
    sections = [
        ("📖 Basic Usage", demo_basic_usage),
        ("🎮 Interactive Controls", demo_interactive_controls),
        ("🔗 Fusion Strategies", demo_fusion_strategies),
        ("🎯 Example Scenarios", demo_example_scenarios),
        ("📺 Understanding Output", demo_understanding_output),
        ("🔧 Troubleshooting", demo_troubleshooting),
        ("💻 System Requirements", demo_system_requirements)
    ]
    
    try:
        for title, func in sections:
            print(f"\n{title}")
            print("=" * len(title))
            func()
            
            # Pause between sections
            input("\nPress ENTER to continue...")
            print()
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Thanks for watching!")
    
    print("\n🎉 Demo Complete!")
    print("\n🚀 Ready to try the system? Run:")
    print("   python multimodal_emotion_inference.py")
    print("\n🧪 Want to test first? Run:")
    print("   python test_multimodal_system.py")
    print("\n📖 Need help? Check:")
    print("   README_multimodal.md")

if __name__ == "__main__":
    main()
