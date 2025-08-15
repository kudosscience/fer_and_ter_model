#!/usr/bin/env python3
"""
Voice-based Textual Emotion Recognition (TER) Inference Script

This script captures voice input, transcribes it using speech recognition,
and predicts emotions using the trained DistilBERT TER model.

Features:
- Real-time voice capture and transcription
- Emotion prediction using the TER model
- Support for both local model files and pre-trained models
- Confidence scores and probability distributions
- Interactive mode for continuous voice input
- Support for different audio input devices

Author: Henry Ward
Date: August 2025
"""

import argparse
import os
import sys
import pickle
import re
import warnings
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

# Audio and speech recognition
import speech_recognition as sr
import pyaudio
import wave

# Machine Learning and NLP
import torch
import torch.nn.functional as F
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Utilities
import time
import threading
from collections import deque

# Suppress warnings
warnings.filterwarnings('ignore')

class VoiceTERInference:
    """
    Voice-based Textual Emotion Recognition inference system
    """
    
    def __init__(self, model_path: str = None, device: str = 'auto', max_length: int = 128):
        """
        Initialize the Voice TER inference system
        
        Args:
            model_path: Path to the trained model directory or file
            device: Device to use ('auto', 'cpu', 'cuda')
            max_length: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.max_length = max_length
        self.device = self._setup_device(device)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.emotion_labels = []
        
        # Audio configuration
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # History tracking
        self.prediction_history = deque(maxlen=50)
        
        # Load model
        self._load_model()
        self._initialize_audio()
        
        print(f"✅ Voice TER Inference initialized successfully!")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Emotions: {', '.join(self.emotion_labels)}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("💻 Using CPU")
        elif device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        
        return torch.device(device)
    
    def _load_model(self):
        """Load the trained TER model and associated components"""
        print("🔄 Loading TER model...")
        
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load from specified path
                model_dir = self.model_path
            else:
                # Try to find model in current directory
                potential_paths = [
                    './models/ter_distilbert_model',
                    './ter_distilbert_model'
                ]
                
                model_dir = None
                for path in potential_paths:
                    if os.path.exists(path):
                        model_dir = path
                        break
                
                if model_dir is None:
                    print("⚠️  No local model found. Using pre-trained DistilBERT...")
                    self._load_pretrained_model()
                    return
            
            # Load model components
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            
            # Load label encoder
            label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                self.emotion_labels = self.label_encoder.classes_.tolist()
            else:
                print("⚠️  Label encoder not found. Using default emotions...")
                self._setup_default_labels()
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded from: {model_dir}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("🔄 Falling back to pre-trained model...")
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load a pre-trained DistilBERT model for emotion classification"""
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            # Note: This would need a pre-trained emotion classification model
            # For now, we'll use the base model and setup default labels
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', 
                num_labels=7
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            self._setup_default_labels()
            print("✅ Pre-trained model loaded")
            
        except Exception as e:
            print(f"❌ Error loading pre-trained model: {str(e)}")
            sys.exit(1)
    
    def _setup_default_labels(self):
        """Setup default emotion labels"""
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Create a simple label encoder
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.emotion_labels)
    
    def _initialize_audio(self):
        """Initialize audio components for speech recognition"""
        try:
            # Test microphone availability
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            print("🎤 Adjusting microphone for ambient noise... (please be quiet)")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
            print("✅ Audio system initialized")
            
        except Exception as e:
            print(f"⚠️  Audio initialization warning: {str(e)}")
            print("🎤 Microphone may not be available for voice input")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def predict_emotion(self, text: str, return_all_probs: bool = True) -> Dict:
        """
        Predict emotion for given text
        
        Args:
            text: Input text to analyze
            return_all_probs: Whether to return probabilities for all emotions
            
        Returns:
            Dictionary containing prediction results
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text.strip():
            return {
                'text': text,
                'cleaned_text': cleaned_text,
                'predicted_emotion': 'neutral',
                'confidence': 0.0,
                'error': 'Empty text after cleaning'
            }
        
        try:
            # Tokenize
            encoding = self.tokenizer(
                cleaned_text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
            # Get probabilities
            probabilities = F.softmax(logits, dim=1)[0]
            
            # Get predicted class
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            predicted_emotion = self.emotion_labels[predicted_class_idx]
            confidence = probabilities[predicted_class_idx].item()
            
            result = {
                'text': text,
                'cleaned_text': cleaned_text,
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_all_probs:
                all_probabilities = {}
                for i, emotion in enumerate(self.emotion_labels):
                    all_probabilities[emotion] = probabilities[i].item()
                result['all_probabilities'] = all_probabilities
            
            return result
            
        except Exception as e:
            return {
                'text': text,
                'cleaned_text': cleaned_text,
                'predicted_emotion': 'neutral',
                'confidence': 0.0,
                'error': f'Prediction error: {str(e)}'
            }
    
    def transcribe_audio(self, timeout: float = 5.0, phrase_time_limit: float = 5.0) -> Optional[str]:
        """
        Transcribe audio from microphone
        
        Args:
            timeout: Seconds to wait for phrase to start
            phrase_time_limit: Seconds for the phrase to complete
            
        Returns:
            Transcribed text or None if failed
        """
        if self.microphone is None:
            print("❌ Microphone not available")
            return None
        
        try:
            print("🎤 Listening... (speak now)")
            
            # Listen for audio
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            print("🔄 Transcribing...")
            
            # Transcribe using Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"📝 Transcribed: '{text}'")
                return text
                
            except sr.UnknownValueError:
                print("❌ Could not understand audio")
                return None
                
            except sr.RequestError as e:
                print(f"❌ Speech recognition error: {e}")
                return None
                
        except sr.WaitTimeoutError:
            print("⏰ No speech detected within timeout")
            return None
            
        except Exception as e:
            print(f"❌ Audio capture error: {str(e)}")
            return None
    
    def voice_to_emotion(self, timeout: float = 5.0, phrase_time_limit: float = 5.0) -> Optional[Dict]:
        """
        Complete pipeline: voice → transcription → emotion prediction
        
        Args:
            timeout: Seconds to wait for phrase to start
            phrase_time_limit: Seconds for the phrase to complete
            
        Returns:
            Emotion prediction results or None if failed
        """
        # Transcribe audio
        text = self.transcribe_audio(timeout, phrase_time_limit)
        
        if text is None:
            return None
        
        # Predict emotion
        result = self.predict_emotion(text)
        
        # Add to history
        self.prediction_history.append(result)
        
        return result
    
    def interactive_mode(self):
        """
        Interactive mode for continuous voice emotion recognition
        """
        print("\n🎯 Interactive Voice Emotion Recognition")
        print("=" * 50)
        print("Commands:")
        print("  - Press ENTER to start voice capture")
        print("  - Type 'text: <your text>' to analyze typed text")
        print("  - Type 'history' to see recent predictions")
        print("  - Type 'stats' to see emotion statistics")
        print("  - Type 'quit' or 'exit' to stop")
        print("=" * 50)
        
        session_count = 0
        
        try:
            while True:
                user_input = input(f"\n[{session_count}] Press ENTER for voice or type command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._show_statistics()
                    continue
                
                elif user_input.lower().startswith('text:'):
                    # Analyze typed text
                    text = user_input[5:].strip()
                    if text:
                        result = self.predict_emotion(text)
                        self._display_result(result)
                        session_count += 1
                    else:
                        print("❌ Please provide text after 'text:'")
                    continue
                
                elif user_input == '':
                    # Voice capture mode
                    print(f"\n🎤 Session {session_count + 1}: Voice Capture")
                    print("-" * 30)
                    
                    result = self.voice_to_emotion(timeout=10.0, phrase_time_limit=10.0)
                    
                    if result:
                        self._display_result(result)
                        session_count += 1
                    else:
                        print("❌ Voice capture failed")
                
                else:
                    print("❌ Unknown command. Press ENTER for voice or type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error in interactive mode: {str(e)}")
        
        finally:
            print(f"\n📊 Session Summary: {session_count} analyses completed")
            if self.prediction_history:
                self._show_statistics()
    
    def _display_result(self, result: Dict):
        """Display emotion prediction result in a formatted way"""
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n📝 Text: '{result['text']}'")
        print(f"🎯 Emotion: {result['predicted_emotion'].upper()}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        
        if 'all_probabilities' in result:
            print(f"\n📈 All Emotions:")
            sorted_probs = sorted(
                result['all_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for emotion, prob in sorted_probs:
                bar_length = int(prob * 20)  # Scale to 20 characters
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {emotion:>8}: {bar} {prob:.3f}")
    
    def _show_history(self, limit: int = 10):
        """Show recent prediction history"""
        if not self.prediction_history:
            print("📭 No prediction history available")
            return
        
        print(f"\n📚 Recent Predictions (last {min(limit, len(self.prediction_history))}):")
        print("-" * 60)
        
        recent = list(self.prediction_history)[-limit:]
        
        for i, result in enumerate(recent, 1):
            timestamp = result.get('timestamp', 'Unknown')
            text = result['text'][:30] + "..." if len(result['text']) > 30 else result['text']
            emotion = result['predicted_emotion']
            confidence = result['confidence']
            
            print(f"{i:2d}. [{timestamp[:19]}] '{text}' → {emotion.upper()} ({confidence:.3f})")
    
    def _show_statistics(self):
        """Show emotion statistics from prediction history"""
        if not self.prediction_history:
            print("📭 No data for statistics")
            return
        
        # Count emotions
        emotion_counts = {}
        total_confidence = 0
        
        for result in self.prediction_history:
            if 'error' not in result:
                emotion = result['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence += result['confidence']
        
        if not emotion_counts:
            print("📭 No valid predictions for statistics")
            return
        
        total_predictions = len(self.prediction_history)
        avg_confidence = total_confidence / total_predictions
        
        print(f"\n📊 Session Statistics:")
        print(f"   Total Predictions: {total_predictions}")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"\n🎭 Emotion Distribution:")
        
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, count in sorted_emotions:
            percentage = (count / total_predictions) * 100
            bar_length = int(percentage / 5)  # Scale to 20 characters max
            bar = "█" * bar_length
            print(f"   {emotion:>8}: {bar:<20} {count:2d} ({percentage:5.1f}%)")
    
    def process_text_file(self, file_path: str, output_path: str = None):
        """
        Process a text file with multiple lines/sentences
        
        Args:
            file_path: Path to input text file
            output_path: Path to save results (optional)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            results = []
            
            print(f"📁 Processing file: {file_path}")
            print(f"📄 Lines to process: {len(lines)}")
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    print(f"\n[{i}/{len(lines)}] Processing: '{line[:50]}...'")
                    result = self.predict_emotion(line)
                    results.append(result)
                    
                    emotion = result['predicted_emotion']
                    confidence = result['confidence']
                    print(f"   → {emotion.upper()} ({confidence:.3f})")
            
            # Save results if output path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Results saved to: {output_path}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            return None


def main():
    """Main function to run the voice TER inference script"""
    parser = argparse.ArgumentParser(
        description="Voice-based Textual Emotion Recognition using DistilBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voice_ter_inference.py                              # Interactive mode
  python voice_ter_inference.py --model ./my_model          # Use specific model
  python voice_ter_inference.py --device cpu                # Force CPU usage
  python voice_ter_inference.py --text "I am very happy!"   # Analyze text directly
  python voice_ter_inference.py --file input.txt            # Process text file
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default=None,
        help='Path to trained model directory (default: auto-detect)'
    )
    
    parser.add_argument(
        '--device', 
        choices=['auto', 'cpu', 'cuda'], 
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    
    parser.add_argument(
        '--text', 
        type=str, 
        default=None,
        help='Analyze specific text and exit'
    )
    
    parser.add_argument(
        '--file', 
        type=str, 
        default=None,
        help='Process text file and exit'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output file for batch processing results'
    )
    
    parser.add_argument(
        '--max-length', 
        type=int, 
        default=128,
        help='Maximum sequence length for tokenization (default: 128)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🎙️  Voice-based Textual Emotion Recognition")
    print("=" * 50)
    
    try:
        # Initialize the inference system
        ter_system = VoiceTERInference(
            model_path=args.model,
            device=args.device,
            max_length=args.max_length
        )
        
        if args.text:
            # Analyze specific text
            print(f"\n🔍 Analyzing text: '{args.text}'")
            result = ter_system.predict_emotion(args.text)
            ter_system._display_result(result)
            
        elif args.file:
            # Process text file
            if not os.path.exists(args.file):
                print(f"❌ File not found: {args.file}")
                sys.exit(1)
            
            ter_system.process_text_file(args.file, args.output)
            
        else:
            # Interactive mode
            ter_system.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
