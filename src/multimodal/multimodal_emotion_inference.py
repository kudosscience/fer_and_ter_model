#!/usr/bin/env python3
"""
Multimodal Emotion Recognition System
Combines Facial Expression Recognition (FER) and Textual Emotion Recognition (TER)
for simultaneous emotion detection from camera and microphone input.

This script integrates:
- Real-time facial expression recognition using camera input
- Voice-based textual emotion recognition using microphone input
- Multimodal emotion fusion and analysis
- Interactive GUI with live feeds and emotion displays

Features:
- Simultaneous FER and TER processing
- Real-time emotion fusion and confidence scoring
- Interactive controls for voice capture
- Save functionality for frames and audio
- Statistics and history tracking
- Configurable fusion strategies

Author: Generated for FER and TER Model Project
Date: August 2025
"""

import argparse
import os
import sys
import pickle
import re
import warnings
from typing import Dict, Tuple, Optional, List, Union
import json
from datetime import datetime
import threading
import queue
import time
from collections import deque, defaultdict

# Computer Vision and Image Processing
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Audio and Speech Recognition
import speech_recognition as sr
import pyaudio
import wave

# Machine Learning and Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants for multimodal fusion
FUSION_WEIGHTS = {
    'facial': 0.6,      # Weight for facial emotion
    'textual': 0.4      # Weight for textual emotion
}

CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to consider a prediction
UPDATE_INTERVAL = 0.1  # Update interval in seconds


class EmotionCNN(nn.Module):
    """
    Convolutional Neural Network for Facial Expression Recognition
    Architecture matches the model defined in the Jupyter notebook
    """
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Second conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class MultimodalEmotionInference:
    """
    Multimodal Emotion Recognition System combining FER and TER
    """
    
    def __init__(self, 
                 fer_model_path: str = None,
                 ter_model_path: str = None,
                 device: str = 'auto',
                 camera_id: int = 0,
                 fusion_strategy: str = 'weighted_average'):
        """
        Initialize the multimodal emotion recognition system
        
        Args:
            fer_model_path: Path to the FER model file
            ter_model_path: Path to the TER model directory
            device: Device to use ('auto', 'cpu', 'cuda')
            camera_id: Camera ID for video capture
            fusion_strategy: Strategy for combining emotions ('weighted_average', 'confidence_based')
        """
        # Common setup
        self.device = self._setup_device(device)
        self.fusion_strategy = fusion_strategy
        
        # Emotion mappings (standardized across both models)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_indices = {emotion: i for i, emotion in enumerate(self.emotion_labels)}
        
        # FER-specific mappings
        self.fer_emotion_labels = {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        
        self.emotion_emojis = {
            'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
            'sad': '😢', 'surprise': '😲', 'neutral': '😐'
        }
        
        self.emotion_colors = {
            'angry': (0, 0, 255),     # Red
            'disgust': (0, 128, 0),   # Green
            'fear': (128, 0, 128),    # Purple
            'happy': (0, 255, 255),   # Yellow
            'sad': (255, 0, 0),       # Blue
            'surprise': (0, 165, 255), # Orange
            'neutral': (128, 128, 128) # Gray
        }
        
        # Initialize components
        self.fer_model = None
        self.ter_model = None
        self.ter_tokenizer = None
        self.ter_label_encoder = None
        
        # Camera and audio setup
        self.camera_id = camera_id
        self.cap = None
        self.face_cascade = None
        self.recognizer = None
        self.microphone = None
        
        # Threading and queues for multimodal processing
        self.audio_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.running = False
        self.audio_thread = None
        
        # State tracking
        self.current_fer_emotion = None
        self.current_ter_emotion = None
        self.current_fused_emotion = None
        self.fer_confidence = 0.0
        self.ter_confidence = 0.0
        self.fused_confidence = 0.0
        
        # History and statistics
        self.prediction_history = deque(maxlen=100)
        self.emotion_counts = defaultdict(int)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # UI state
        self.fullscreen = False
        self.show_ter_panel = True
        self.voice_capture_active = False
        
        # Load models
        self._load_fer_model(fer_model_path)
        self._load_ter_model(ter_model_path)
        self._initialize_camera()
        self._initialize_audio()
        
        print(f"✅ Multimodal Emotion Recognition System initialized!")
        print(f"📱 Device: {self.device}")
        print(f"🎭 Emotions: {', '.join(self.emotion_labels)}")
    
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
    
    def _load_fer_model(self, model_path: str):
        """Load the FER model"""
        print("🔄 Loading FER model...")
        
        if model_path is None:
            model_path = 'models/fer2013_final_model.pth'
        
        if not os.path.exists(model_path):
            print(f"❌ FER model not found: {model_path}")
            print("⚠️  FER functionality will be disabled")
            return
        
        try:
            # Create model instance
            self.fer_model = EmotionCNN(num_classes=7, dropout_rate=0.5)
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✅ FER checkpoint loaded with accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
            else:
                state_dict = checkpoint
            
            self.fer_model.load_state_dict(state_dict)
            self.fer_model.eval()
            self.fer_model.to(self.device)
            
            # Define image preprocessing transforms
            self.fer_transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            print(f"✅ FER model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading FER model: {str(e)}")
            self.fer_model = None
    
    def _load_ter_model(self, model_path: str):
        """Load the TER model"""
        print("🔄 Loading TER model...")
        
        try:
            if model_path and os.path.exists(model_path):
                model_dir = model_path
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
                    print("⚠️  No local TER model found. Using pre-trained DistilBERT...")
                    self._load_pretrained_ter_model()
                    return
            
            # Load model components
            self.ter_tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            self.ter_model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            
            # Load label encoder
            label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.ter_label_encoder = pickle.load(f)
            else:
                print("⚠️  TER label encoder not found. Using default emotions...")
                self._setup_default_ter_labels()
            
            self.ter_model.to(self.device)
            self.ter_model.eval()
            
            print(f"✅ TER model loaded from: {model_dir}")
            
        except Exception as e:
            print(f"❌ Error loading TER model: {str(e)}")
            print("🔄 Falling back to pre-trained model...")
            self._load_pretrained_ter_model()
    
    def _load_pretrained_ter_model(self):
        """Load a pre-trained DistilBERT model for emotion classification"""
        try:
            self.ter_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.ter_model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', 
                num_labels=7
            )
            
            self.ter_model.to(self.device)
            self.ter_model.eval()
            
            self._setup_default_ter_labels()
            print("✅ Pre-trained TER model loaded")
            
        except Exception as e:
            print(f"❌ Error loading pre-trained TER model: {str(e)}")
            self.ter_model = None
    
    def _setup_default_ter_labels(self):
        """Setup default TER emotion labels"""
        from sklearn.preprocessing import LabelEncoder
        self.ter_label_encoder = LabelEncoder()
        self.ter_label_encoder.fit(self.emotion_labels)
    
    def _initialize_camera(self):
        """Initialize camera and face detection"""
        if self.fer_model is None:
            print("⚠️  Skipping camera initialization (FER model not loaded)")
            return
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize face detector
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            print(f"✅ Camera {self.camera_id} initialized successfully")
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {str(e)}")
            self.cap = None
            self.face_cascade = None
    
    def _initialize_audio(self):
        """Initialize audio components for speech recognition"""
        if self.ter_model is None:
            print("⚠️  Skipping audio initialization (TER model not loaded)")
            return
        
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            print("🎤 Adjusting microphone for ambient noise... (please be quiet)")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
            print("✅ Audio system initialized")
            
        except Exception as e:
            print(f"⚠️  Audio initialization warning: {str(e)}")
            self.recognizer = None
            self.microphone = None
    
    def _preprocess_face(self, face_img):
        """Preprocess face image for FER"""
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        pil_img = Image.fromarray(face_img)
        tensor_img = self.fer_transform(pil_img)
        tensor_img = tensor_img.unsqueeze(0)
        
        return tensor_img.to(self.device)
    
    def _predict_fer_emotion(self, face_tensor):
        """Predict emotion from face tensor"""
        if self.fer_model is None:
            return None, None
        
        with torch.no_grad():
            outputs = self.fer_model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence_scores = probabilities.cpu().numpy()[0]
        
        # Convert FER indices to standardized emotion labels
        fer_emotion = self.fer_emotion_labels[predicted_class]
        
        return fer_emotion, confidence_scores[predicted_class]
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _predict_ter_emotion(self, text: str):
        """Predict emotion from text"""
        if self.ter_model is None or not text.strip():
            return None, None
        
        try:
            cleaned_text = self._clean_text(text)
            
            if not cleaned_text.strip():
                return None, None
            
            # Tokenize
            encoding = self.ter_tokenizer(
                cleaned_text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.ter_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=1)[0]
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            
            # Convert to standardized emotion label
            if self.ter_label_encoder:
                predicted_emotion = self.ter_label_encoder.classes_[predicted_class_idx]
            else:
                predicted_emotion = self.emotion_labels[predicted_class_idx]
            
            confidence = probabilities[predicted_class_idx].item()
            
            return predicted_emotion, confidence
            
        except Exception as e:
            print(f"❌ TER prediction error: {str(e)}")
            return None, None
    
    def _fuse_emotions(self, fer_emotion: str, fer_confidence: float, 
                      ter_emotion: str, ter_confidence: float) -> Tuple[str, float]:
        """
        Fuse emotions from FER and TER models
        
        Args:
            fer_emotion: Emotion from facial recognition
            fer_confidence: Confidence of FER prediction
            ter_emotion: Emotion from text recognition
            ter_confidence: Confidence of TER prediction
            
        Returns:
            tuple: (fused_emotion, fused_confidence)
        """
        if self.fusion_strategy == 'weighted_average':
            return self._weighted_average_fusion(fer_emotion, fer_confidence, ter_emotion, ter_confidence)
        elif self.fusion_strategy == 'confidence_based':
            return self._confidence_based_fusion(fer_emotion, fer_confidence, ter_emotion, ter_confidence)
        else:
            # Default to confidence-based
            return self._confidence_based_fusion(fer_emotion, fer_confidence, ter_emotion, ter_confidence)
    
    def _weighted_average_fusion(self, fer_emotion: str, fer_confidence: float, 
                                ter_emotion: str, ter_confidence: float) -> Tuple[str, float]:
        """Weighted average fusion strategy"""
        # Handle None confidence values
        if fer_confidence is None:
            fer_confidence = 0.0
        if ter_confidence is None:
            ter_confidence = 0.0
            
        if fer_emotion is None and ter_emotion is None:
            return 'neutral', 0.0
        elif fer_emotion is None:
            return ter_emotion, ter_confidence
        elif ter_emotion is None:
            return fer_emotion, fer_confidence
        
        # Both emotions are available
        fer_weight = FUSION_WEIGHTS['facial']
        ter_weight = FUSION_WEIGHTS['textual']
        
        # If emotions agree, boost confidence
        if fer_emotion == ter_emotion:
            fused_confidence = (fer_confidence * fer_weight + ter_confidence * ter_weight) * 1.2
            fused_confidence = min(fused_confidence, 1.0)  # Cap at 1.0
            return fer_emotion, fused_confidence
        
        # If emotions disagree, choose the more confident one but reduce confidence
        if fer_confidence * fer_weight > ter_confidence * ter_weight:
            fused_confidence = fer_confidence * fer_weight * 0.8  # Reduce confidence due to disagreement
            return fer_emotion, fused_confidence
        else:
            fused_confidence = ter_confidence * ter_weight * 0.8
            return ter_emotion, fused_confidence
    
    def _confidence_based_fusion(self, fer_emotion: str, fer_confidence: float, 
                                ter_emotion: str, ter_confidence: float) -> Tuple[str, float]:
        """Confidence-based fusion strategy"""
        # Handle None confidence values
        if fer_confidence is None:
            fer_confidence = 0.0
        if ter_confidence is None:
            ter_confidence = 0.0
            
        if fer_emotion is None and ter_emotion is None:
            return 'neutral', 0.0
        elif fer_emotion is None:
            return ter_emotion, ter_confidence
        elif ter_emotion is None:
            return fer_emotion, fer_confidence
        
        # Both emotions are available - choose the more confident one
        if fer_confidence > ter_confidence:
            # Boost confidence if emotions agree
            boost = 1.1 if fer_emotion == ter_emotion else 0.9
            return fer_emotion, min(fer_confidence * boost, 1.0)
        else:
            boost = 1.1 if fer_emotion == ter_emotion else 0.9
            return ter_emotion, min(ter_confidence * boost, 1.0)
    
    def _transcribe_audio(self, timeout: float = 3.0) -> Optional[str]:
        """Transcribe audio from microphone"""
        if self.microphone is None or self.recognizer is None:
            return None
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3.0)
            
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except (sr.UnknownValueError, sr.RequestError):
                return None
                
        except sr.WaitTimeoutError:
            return None
        except Exception:
            return None
    
    def _audio_processing_thread(self):
        """Background thread for continuous audio processing"""
        while self.running:
            try:
                if self.voice_capture_active:
                    text = self._transcribe_audio(timeout=2.0)
                    if text:
                        emotion, confidence = self._predict_ter_emotion(text)
                        if emotion and confidence > CONFIDENCE_THRESHOLD:
                            self.audio_queue.put({
                                'text': text,
                                'emotion': emotion,
                                'confidence': confidence,
                                'timestamp': time.time()
                            })
                time.sleep(0.1)
            except Exception as e:
                print(f"Audio processing error: {e}")
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _draw_fer_predictions(self, frame, faces, predictions):
        """Draw FER predictions on frame"""
        for (x, y, w, h), (emotion, confidence) in zip(faces, predictions):
            if emotion is None or confidence is None:
                continue
                
            color = self.emotion_colors[emotion]
            
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare text
            emoji = self.emotion_emojis[emotion]
            label = f"{emotion.title()} {emoji}"
            label_confidence = f"{confidence:.2%}"
            
            # Calculate text position
            text_y = y - 10 if y > 40 else y + h + 25
            
            # Draw text background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, text_y - text_h - 5), (x + text_w, text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            cv2.putText(frame, label_confidence, (x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
    
    def _draw_ui_panels(self, frame):
        """Draw UI panels with multimodal information"""
        height, width = frame.shape[:2]
        
        # Main info panel (top-left)
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # FPS and device info
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        device_text = f"Device: {self.device.type.upper()}"
        cv2.putText(frame, device_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current emotions
        fer_text = f"FER: {self.current_fer_emotion or 'None'}"
        ter_text = f"TER: {self.current_ter_emotion or 'None'}"
        fused_text = f"Fused: {self.current_fused_emotion or 'None'}"
        
        cv2.putText(frame, fer_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(frame, ter_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        cv2.putText(frame, fused_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
        
        # TER panel (bottom-left)
        if self.show_ter_panel:
            panel_y = height - 150
            cv2.rectangle(frame, (10, panel_y), (400, height - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, panel_y), (400, height - 10), (255, 255, 255), 2)
            
            # Voice capture status
            status_color = (0, 255, 0) if self.voice_capture_active else (0, 0, 255)
            status_text = "LISTENING" if self.voice_capture_active else "PRESS 'V' TO TALK"
            cv2.putText(frame, status_text, (20, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, status_color, 2)
            
            # Recent TER prediction
            if hasattr(self, 'last_ter_text') and self.last_ter_text:
                text_display = self.last_ter_text[:40] + "..." if len(self.last_ter_text) > 40 else self.last_ter_text
                cv2.putText(frame, f"Text: {text_display}", (20, panel_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # TER confidence
            if self.ter_confidence is not None and self.ter_confidence > 0:
                conf_text = f"TER Confidence: {self.ter_confidence:.2%}"
                cv2.putText(frame, conf_text, (20, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (255, 255, 255), 1)
        
        # Fusion panel (bottom-right)
        if self.current_fused_emotion and self.fused_confidence is not None:
            panel_x = width - 200
            panel_y = height - 100
            cv2.rectangle(frame, (panel_x, panel_y), (width - 10, height - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (panel_x, panel_y), (width - 10, height - 10), (255, 255, 100), 2)
            
            # Fused emotion
            emoji = self.emotion_emojis.get(self.current_fused_emotion, '❓')
            fused_display = f"{self.current_fused_emotion.title()} {emoji}"
            cv2.putText(frame, fused_display, (panel_x + 10, panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
            
            # Fused confidence
            conf_text = f"Conf: {self.fused_confidence:.2%}"
            cv2.putText(frame, conf_text, (panel_x + 10, panel_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls (top-right)
        controls = [
            "Controls:",
            "Q - Quit",
            "V - Toggle Voice",
            "S - Save Frame",
            "T - Toggle TER Panel",
            "F - Fullscreen",
            "H - Show History",
            "P - Show Statistics"
        ]
        
        for i, control in enumerate(controls):
            y_pos = 30 + (i * 18)
            cv2.putText(frame, control, (width - 180, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
    
    def _save_frame(self, frame):
        """Save current frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multimodal_emotion_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Frame saved as {filename}")
    
    def _show_history(self):
        """Display emotion prediction history"""
        if not self.prediction_history:
            print("📭 No prediction history available")
            return
        
        print(f"\n📚 Recent Multimodal Predictions (last 10):")
        print("-" * 80)
        
        recent = list(self.prediction_history)[-10:]
        
        for i, result in enumerate(recent, 1):
            timestamp = result.get('timestamp', 'Unknown')
            fer = result.get('fer_emotion', 'None')
            ter = result.get('ter_emotion', 'None')
            fused = result.get('fused_emotion', 'None')
            fused_conf = result.get('fused_confidence', 0.0)
            
            # Handle None values safely
            if timestamp is None:
                timestamp = 'Unknown'
            if fer is None:
                fer = 'None'
            if ter is None:
                ter = 'None'
            if fused is None:
                fused = 'None'
            if fused_conf is None:
                fused_conf = 0.0
            
            # Safely slice timestamp
            timestamp_display = timestamp[:19] if len(str(timestamp)) >= 19 else str(timestamp)
            
            print(f"{i:2d}. [{timestamp_display}] FER:{fer:>8} | TER:{ter:>8} | Fused:{fused:>8} ({fused_conf:.3f})")
    
    def _show_statistics(self):
        """Show emotion statistics"""
        if not self.emotion_counts:
            print("📭 No data for statistics")
            return
        
        total = sum(self.emotion_counts.values())
        
        print("\n📊 Session Statistics:")
        print(f"   Total Predictions: {total}")
        print("\n🎭 Emotion Distribution:")
        
        sorted_emotions = sorted(self.emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, count in sorted_emotions:
            percentage = (count / total) * 100
            bar_length = int(percentage / 5)
            bar = "█" * bar_length
            emoji = self.emotion_emojis.get(emotion, '❓')
            print(f"   {emotion:>8} {emoji}: {bar:<20} {count:2d} ({percentage:5.1f}%)")
    
    def run(self):
        """Main multimodal emotion recognition loop"""
        if self.cap is None and self.microphone is None:
            print("❌ Neither camera nor microphone is available")
            return
        
        print("🎯 Starting Multimodal Emotion Recognition...")
        print("Press 'V' to toggle voice capture, 'Q' to quit")
        
        # Start audio processing thread
        self.running = True
        if self.microphone is not None:
            self.audio_thread = threading.Thread(target=self._audio_processing_thread, daemon=True)
            self.audio_thread.start()
        
        try:
            while True:
                frame = None
                
                # Camera processing
                if self.cap is not None:
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)  # Mirror effect
                        
                        # Face detection and FER
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                        )
                        
                        # Process faces
                        fer_predictions = []
                        self.current_fer_emotion = None
                        self.fer_confidence = 0.0
                        
                        for (x, y, w, h) in faces:
                            face_roi = gray[y:y+h, x:x+w]
                            face_tensor = self._preprocess_face(face_roi)
                            emotion, confidence = self._predict_fer_emotion(face_tensor)
                            
                            if emotion and confidence > CONFIDENCE_THRESHOLD:
                                fer_predictions.append((emotion, confidence))
                                # Use the most confident face for current emotion
                                if confidence > self.fer_confidence:
                                    self.current_fer_emotion = emotion
                                    self.fer_confidence = confidence
                        
                        # Draw FER predictions
                        if fer_predictions:
                            self._draw_fer_predictions(frame, faces, fer_predictions)
                else:
                    # Create a black frame if no camera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Process audio queue
                self.current_ter_emotion = None
                self.ter_confidence = 0.0
                
                while not self.audio_queue.empty():
                    try:
                        audio_result = self.audio_queue.get_nowait()
                        self.current_ter_emotion = audio_result['emotion']
                        self.ter_confidence = audio_result['confidence']
                        self.last_ter_text = audio_result['text']
                        print(f"🎤 '{audio_result['text']}' → {audio_result['emotion']} ({audio_result['confidence']:.3f})")
                    except queue.Empty:
                        break
                
                # Fusion
                if self.current_fer_emotion or self.current_ter_emotion:
                    self.current_fused_emotion, self.fused_confidence = self._fuse_emotions(
                        self.current_fer_emotion, self.fer_confidence,
                        self.current_ter_emotion, self.ter_confidence
                    )
                    
                    # Update statistics
                    if self.current_fused_emotion:
                        self.emotion_counts[self.current_fused_emotion] += 1
                    
                    # Add to history
                    prediction_result = {
                        'timestamp': datetime.now().isoformat(),
                        'fer_emotion': self.current_fer_emotion,
                        'fer_confidence': self.fer_confidence,
                        'ter_emotion': self.current_ter_emotion,
                        'ter_confidence': self.ter_confidence,
                        'fused_emotion': self.current_fused_emotion,
                        'fused_confidence': self.fused_confidence
                    }
                    self.prediction_history.append(prediction_result)
                
                # Draw UI
                if frame is not None:
                    self._draw_ui_panels(frame)
                    self._update_fps()
                    
                    # Display frame
                    if self.fullscreen:
                        cv2.namedWindow('Multimodal Emotion Recognition', cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty('Multimodal Emotion Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.namedWindow('Multimodal Emotion Recognition', cv2.WINDOW_NORMAL)
                    
                    cv2.imshow('Multimodal Emotion Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self.voice_capture_active = not self.voice_capture_active
                    status = "ON" if self.voice_capture_active else "OFF"
                    print(f"🎤 Voice capture: {status}")
                elif key == ord('s') and frame is not None:
                    self._save_frame(frame)
                elif key == ord('t'):
                    self.show_ter_panel = not self.show_ter_panel
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    if not self.fullscreen:
                        cv2.setWindowProperty('Multimodal Emotion Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif key == ord('h'):
                    self._show_history()
                elif key == ord('p'):  # Print statistics
                    self._show_statistics()
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error during inference: {e}")
        finally:
            # Cleanup
            self.running = False
            if self.audio_thread:
                self.audio_thread.join(timeout=2.0)
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("🔄 Cleanup completed")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Multimodal Emotion Recognition System (FER + TER)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multimodal_emotion_inference.py                                    # Default settings
  python multimodal_emotion_inference.py --fer_model ./my_fer_model.pth     # Custom FER model
  python multimodal_emotion_inference.py --ter_model ./my_ter_model         # Custom TER model
  python multimodal_emotion_inference.py --device cpu                       # Force CPU usage
  python multimodal_emotion_inference.py --fusion confidence_based          # Change fusion strategy
        """
    )
    
    parser.add_argument(
        '--fer_model', 
        type=str, 
        default=None,
        help='Path to FER model file (default: models/fer2013_final_model.pth)'
    )
    
    parser.add_argument(
        '--ter_model', 
        type=str, 
        default=None,
        help='Path to TER model directory (default: auto-detect)'
    )
    
    parser.add_argument(
        '--device', 
        choices=['auto', 'cpu', 'cuda'], 
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    
    parser.add_argument(
        '--camera_id', 
        type=int, 
        default=0,
        help='Camera ID for video capture (default: 0)'
    )
    
    parser.add_argument(
        '--fusion', 
        choices=['weighted_average', 'confidence_based'], 
        default='confidence_based',
        help='Fusion strategy for combining emotions (default: confidence_based)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🎭 Multimodal Emotion Recognition System")
    print("=" * 50)
    print("Combining Facial Expression Recognition (FER) and Textual Emotion Recognition (TER)")
    print("=" * 50)
    
    try:
        # Initialize the multimodal system
        system = MultimodalEmotionInference(
            fer_model_path=args.fer_model,
            ter_model_path=args.ter_model,
            device=args.device,
            camera_id=args.camera_id,
            fusion_strategy=args.fusion
        )
        
        # Run the system
        system.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
