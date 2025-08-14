#!/usr/bin/env python3
"""
Demonstration script for Voice-based Textual Emotion Recognition

This script shows various ways to use the voice TER inference system
without requiring actual voice input for testing purposes.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_text_analysis():
    """Demonstrate text-based emotion analysis"""
    print("📝 Text Analysis Demo")
    print("=" * 40)
    
    # Import the VoiceTERInference class
    try:
        from voice_ter_inference import VoiceTERInference
    except ImportError as e:
        print(f"❌ Error importing VoiceTERInference: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements_voice_ter.txt")
        return
    
    # Sample texts representing different emotions
    sample_texts = [
        "I am so excited about this new opportunity! This is the best day ever!",
        "I can't believe they did this to me. I am absolutely furious right now.",
        "I'm really worried about what might happen tomorrow. This is so scary.",
        "That food was absolutely disgusting. I feel sick just thinking about it.",
        "I feel so lonely and sad. Nothing seems to go right anymore.",
        "Wow! I never saw that coming! What an incredible surprise!",
        "The weather is okay today. Nothing particularly special happening.",
        "I love spending time with my family. They make me so happy.",
        "This situation is making me very anxious and fearful.",
        "I am so angry I could scream right now!"
    ]
    
    try:
        # Initialize the TER system
        print("🔄 Initializing TER system...")
        ter_system = VoiceTERInference(device='auto')
        
        print(f"\n🎯 Analyzing {len(sample_texts)} sample texts:")
        print("-" * 60)
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n[{i}] Text: '{text}'")
            
            # Predict emotion
            result = ter_system.predict_emotion(text)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            emotion = result['predicted_emotion']
            confidence = result['confidence']
            
            print(f"🎭 Predicted Emotion: {emotion.upper()}")
            print(f"📊 Confidence: {confidence:.3f}")
            
            # Show top 3 emotions
            if 'all_probabilities' in result:
                sorted_probs = sorted(
                    result['all_probabilities'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                print("🏆 Top 3 emotions:")
                for emotion_name, prob in sorted_probs:
                    print(f"   {emotion_name}: {prob:.3f}")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure all dependencies are installed")
        print("2. Check if you have a trained model in the expected location")
        print("3. Try running with --device cpu if GPU issues occur")

def demo_batch_processing():
    """Demonstrate batch text file processing"""
    print("\n📁 Batch Processing Demo")
    print("=" * 40)
    
    # Create a sample text file
    sample_file = "sample_texts.txt"
    sample_content = """I am so happy today!
This is really making me angry.
I'm scared about the future.
That smells absolutely disgusting.
I feel very sad and lonely.
What a wonderful surprise!
The meeting went fine today.
I love this amazing weather!
This situation is quite frightening.
I'm furious about this decision!"""
    
    try:
        # Write sample file
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        print(f"📝 Created sample file: {sample_file}")
        print("📄 Contents:")
        print("-" * 30)
        
        lines = sample_content.split('\n')
        for i, line in enumerate(lines, 1):
            print(f"{i:2d}. {line}")
        
        print(f"\n💡 To process this file, run:")
        print(f"   python voice_ter_inference.py --file {sample_file} --output results.json")
        
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)
            print(f"🧹 Cleaned up sample file")
        
    except Exception as e:
        print(f"❌ Batch demo error: {str(e)}")

def show_usage_examples():
    """Show comprehensive usage examples"""
    print("\n📚 Usage Examples")
    print("=" * 40)
    
    examples = [
        ("Interactive Voice Mode", 
         "python voice_ter_inference.py",
         "Start interactive mode for voice input and real-time emotion analysis"),
        
        ("Analyze Specific Text", 
         'python voice_ter_inference.py --text "I am feeling great today!"',
         "Analyze a specific text string"),
        
        ("Process Text File", 
         "python voice_ter_inference.py --file input.txt --output results.json",
         "Process all lines in a text file and save results"),
        
        ("Use Specific Model", 
         "python voice_ter_inference.py --model ./my_model",
         "Use a specific trained model directory"),
        
        ("Force CPU Usage", 
         "python voice_ter_inference.py --device cpu",
         "Force CPU usage (useful if GPU has issues)"),
        
        ("Custom Max Length", 
         "python voice_ter_inference.py --max-length 256",
         "Use custom maximum sequence length for tokenization")
    ]
    
    for title, command, description in examples:
        print(f"\n🔹 {title}:")
        print(f"   Command: {command}")
        print(f"   Description: {description}")

def show_interactive_commands():
    """Show interactive mode commands"""
    print("\n🎮 Interactive Mode Commands")
    print("=" * 40)
    
    commands = [
        ("ENTER", "Start voice capture and emotion analysis"),
        ("text: <message>", "Analyze typed text directly"),
        ("history", "Show recent prediction history"),
        ("stats", "Show emotion statistics for current session"),
        ("quit / exit / q", "Exit the application")
    ]
    
    for command, description in commands:
        print(f"📌 {command:<15} - {description}")

def check_system_requirements():
    """Check if system requirements are met"""
    print("\n🔍 System Requirements Check")
    print("=" * 40)
    
    requirements = [
        ("Python", sys.version, "✅"),
        ("Platform", sys.platform, "✅")
    ]
    
    # Check required packages
    packages_to_check = [
        'torch', 'transformers', 'sklearn', 'numpy', 
        'speech_recognition', 'pyaudio'
    ]
    
    for package in packages_to_check:
        try:
            __import__(package)
            requirements.append((f"Package: {package}", "Available", "✅"))
        except ImportError:
            requirements.append((f"Package: {package}", "Missing", "❌"))
    
    # Display results
    for item, status, icon in requirements:
        print(f"{icon} {item:<20}: {status}")
    
    # Check for microphone (basic check)
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        p.terminate()
        requirements.append(("Audio Devices", f"{device_count} found", "✅"))
    except Exception:
        requirements.append(("Audio Devices", "Check manually", "⚠️"))
    
    print(f"\n💡 If any packages are missing, install with:")
    print(f"   pip install -r requirements_voice_ter.txt")

def main():
    """Main demo function"""
    print("🎙️  Voice TER Inference - Demo & Examples")
    print("=" * 50)
    
    # Show system check
    check_system_requirements()
    
    # Show usage examples
    show_usage_examples()
    
    # Show interactive commands
    show_interactive_commands()
    
    # Ask user what they want to demo
    print(f"\n🎯 What would you like to demo?")
    print("1. Text Analysis Demo")
    print("2. Batch Processing Demo")
    print("3. Show Usage Examples (already shown)")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            demo_text_analysis()
        elif choice == "2":
            demo_batch_processing()
        elif choice == "3":
            print("✅ Examples already shown above!")
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Run the script again.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")

if __name__ == "__main__":
    main()
