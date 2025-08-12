#!/usr/bin/env python3
"""
Furhat Multimodal Emotion Recognition System
Combines Facial Expression Recognition (FER) and Textual Emotion Recognition (TER)
for emotion detection using device camera and Furhat robot's microphone.

This script integrates:
- Real-time facial expression recognition using device camera
- Voice-based textual emotion recognition using Furhat's microphone
- Multimodal emotion fusion and analysis
- Furhat robot control for emotional responses

Features:
- Furhat Remote API integration for TER and robot responses
- Device camera integration for FER processing
- Real-time emotion fusion and confidence scoring
- Robot emotional gestures and voice responses
- Interactive controls for voice capture
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
import io
import base64

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

# Furhat Remote API
try:
    from furhat_remote_api import FurhatRemoteAPI
    FURHAT_AVAILABLE = True
except ImportError:
    FURHAT_AVAILABLE = False
    print("⚠️  furhat-remote-api not installed. Install with: pip install furhat-remote-api")

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants for multimodal fusion
FUSION_WEIGHTS = {
    'facial': 0.6,      # Weight for facial emotion
    'textual': 0.4      # Weight for textual emotion
}

# Model performance metrics for formula-based fusion
# These would typically be derived from validation data
MODEL_PERFORMANCE = {
    'fer': {
        'accuracy': 0.73,  # Overall model accuracy
        'recall': {        # Per-emotion recall scores
            'angry': 0.71, 'disgust': 0.68, 'fear': 0.65, 'happy': 0.84,
            'sad': 0.70, 'surprise': 0.75, 'neutral': 0.78
        }
    },
    'ter': {
        'accuracy': 0.79,  # Overall model accuracy  
        'recall': {        # Per-emotion recall scores
            'angry': 0.76, 'disgust': 0.72, 'fear': 0.69, 'happy': 0.87,
            'sad': 0.74, 'surprise': 0.71, 'neutral': 0.82
        }
    }
}

CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to consider a prediction
UPDATE_INTERVAL = 0.1  # Update interval in seconds

# Furhat robot configuration
FURHAT_IP = "localhost"  # Default Furhat IP address
FURHAT_PORT = 54321     # Default Furhat port

# Emotion to gesture mapping for Furhat (will be validated and updated at runtime)
EMOTION_GESTURES = {
    'happy': 'BigSmile',
    'sad': 'Sad',
    'angry': 'ExpressAnger', 
    'surprise': 'Surprised',
    'fear': 'Worried',
    'disgust': 'Disgusted',
    'neutral': None  # No specific gesture for neutral
}

# Emotion to voice responses
EMOTION_RESPONSES = {
    'happy': ["I can see you're feeling happy!", "That's wonderful!", "You look joyful!"],
    'sad': ["I notice you seem sad.", "Is everything alright?", "I sense some sadness."],
    'angry': ["You appear to be upset.", "I can see you're angry.", "What's troubling you?"],
    'surprise': ["You look surprised!", "What a surprise!", "That caught your attention!"],
    'fear': ["You seem worried.", "Is something concerning you?", "I sense some anxiety."],
    'disgust': ["You look disgusted.", "Something doesn't seem right.", "That's unpleasant."],
    'neutral': ["You appear calm.", "Everything seems normal.", "You look composed."]
}


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


class FurhatMultimodalEmotionInference:
    """
    Multimodal Emotion Recognition System using Furhat robot's sensors
    """
    
    def __init__(self, 
                 fer_model_path: str = None,
                 ter_model_path: str = None,
                 device: str = 'auto',
                 furhat_ip: str = FURHAT_IP,
                 camera_id: int = 0,
                 fusion_strategy: str = 'weighted_average',
                 enable_robot_responses: bool = True):
        """
        Initialize the Furhat multimodal emotion recognition system
        
        Args:
            fer_model_path: Path to the FER model file
            ter_model_path: Path to the TER model directory
            device: Device to use ('auto', 'cpu', 'cuda')
            furhat_ip: IP address of the Furhat robot
            camera_id: Camera ID for device camera
            fusion_strategy: Strategy for combining emotions ('weighted_average', 'confidence_based', 'formula_based')
            enable_robot_responses: Whether to enable robot emotional responses
        """
        # Check Furhat availability
        if not FURHAT_AVAILABLE:
            raise ImportError("furhat-remote-api is required. Install with: pip install furhat-remote-api")
        
        # Common setup
        self.device = self._setup_device(device)
        self.fusion_strategy = fusion_strategy
        self.furhat_ip = furhat_ip
        self.enable_robot_responses = enable_robot_responses
        
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
        
        # Furhat robot setup
        self.furhat = None
        self.face_cascade = None
        self.recognizer = None
        
        # Camera setup for local device camera
        self.camera_id = camera_id  # Use parameter instead of default
        self.cap = None
        
        # FER preprocessing transforms
        self.fer_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        
        # Threading and queues for multimodal processing
        self.audio_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.running = False
        self.audio_thread = None
        self.robot_response_thread = None
        
        # State tracking
        self.current_fer_emotion = None
        self.current_ter_emotion = None
        self.current_fused_emotion = None
        self.fer_confidence = 0.0
        self.ter_confidence = 0.0
        self.fused_confidence = 0.0
        self.last_ter_text = ""
        
        # History and statistics
        self.prediction_history = deque(maxlen=100)
        self.emotion_counts = defaultdict(int)
        self.last_robot_response_time = 0
        self.robot_response_cooldown = 3.0  # Seconds between robot responses
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # UI state
        self.fullscreen = False
        self.show_ter_panel = True
        self.voice_capture_active = False
        
        # Load models and initialize connections
        self._load_fer_model(fer_model_path)
        self._load_ter_model(ter_model_path)
        self._initialize_furhat()
        self._initialize_face_detection()
        self._initialize_speech_recognition()
        self._initialize_camera()  # Add camera initialization
        
        print(f"✅ Furhat Multimodal Emotion Recognition System initialized!")
        print(f"🤖 Robot IP: {self.furhat_ip}")
        print(f"� Camera ID: {self.camera_id}")
        print(f"�📱 Device: {self.device}")
        print(f"🎭 Emotions: {', '.join(self.emotion_labels)}")
        print(f"🔄 Fusion Strategy: {self.fusion_strategy}")
    
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
    
    def _initialize_furhat(self):
        """Initialize connection to Furhat robot"""
        try:
            print(f"🤖 Connecting to Furhat robot at {self.furhat_ip}...")
            self.furhat = FurhatRemoteAPI(self.furhat_ip)
            
            # Test connection
            response = self.furhat.get_users()
            print("✅ Furhat robot connected successfully")
            
            # Test and validate available gestures
            self._validate_gestures()
            
            # Set initial robot state
            self.furhat.say(text="Hello! I'm ready for emotion recognition.")
            
            # Get available voices
            voices = self.furhat.get_voices()
            if voices:
                print(f"🗣️  Available voices: {len(voices)}")
            
        except Exception as e:
            print(f"❌ Failed to connect to Furhat robot: {str(e)}")
            print("🔧 Make sure Furhat robot is running and Remote API is enabled")
            self.furhat = None
    
    def _validate_gestures(self):
        """Validate that emotion gestures are available on the robot"""
        if self.furhat is None:
            return
        
        try:
            # Get available gestures from Furhat using the API
            gestures_response = self.furhat.get_gestures()
            
            if gestures_response:
                # The response should be a list of gesture objects with 'name' and 'duration' properties
                available_gesture_names = []
                
                # Handle different response formats
                if hasattr(gestures_response, '__iter__'):
                    for gesture in gestures_response:
                        if hasattr(gesture, 'name'):
                            available_gesture_names.append(gesture.name)
                        elif isinstance(gesture, dict) and 'name' in gesture:
                            available_gesture_names.append(gesture['name'])
                
                print(f"🎭 Available gestures ({len(available_gesture_names)}): {', '.join(available_gesture_names[:10])}{'...' if len(available_gesture_names) > 10 else ''}")
                
                # Check each emotion gesture mapping
                invalid_gestures = []
                valid_gestures = []
                
                for emotion, gesture in EMOTION_GESTURES.items():
                    if gesture:  # Skip None values
                        if gesture in available_gesture_names:
                            valid_gestures.append((emotion, gesture))
                        else:
                            invalid_gestures.append((emotion, gesture))
                
                if invalid_gestures:
                    print("⚠️  Invalid gesture mappings found, trying alternatives:")
                    for emotion, invalid_gesture in invalid_gestures:
                        print(f"   {emotion}: {invalid_gesture} (not available)")
                        
                        # Try to find alternative gestures for emotions
                        alternative = self._find_alternative_gesture(emotion, available_gesture_names)
                        if alternative:
                            EMOTION_GESTURES[emotion] = alternative
                            print(f"   → Using alternative: {alternative}")
                        else:
                            EMOTION_GESTURES[emotion] = None
                            print(f"   → No alternative found, disabled gestures for {emotion}")
                
                if valid_gestures:
                    print(f"✅ Valid gesture mappings:")
                    for emotion, gesture in valid_gestures:
                        print(f"   {emotion}: {gesture}")
                
            else:
                print("⚠️  Could not retrieve gesture list from robot")
                
        except Exception as e:
            print(f"⚠️  Gesture validation failed: {e}")
            print("🔧 Will attempt to use default gestures but they may fail")
    
    def _find_alternative_gesture(self, emotion: str, available_gestures: list) -> str:
        """Find alternative gesture names for emotions based on available gestures"""
        # Common alternative gesture mappings based on typical Furhat gesture names
        emotion_alternatives = {
            'happy': ['BigSmile', 'Smile', 'Happy', 'Joy', 'Pleased'],
            'sad': ['Sad', 'Frown', 'Disappointed', 'Unhappy'],
            'angry': ['Angry', 'Mad', 'Annoyed', 'Frustrated'],
            'surprise': ['Surprised', 'Surprise', 'Shocked', 'Amazed'],
            'fear': ['Worried', 'Scared', 'Afraid', 'Anxious', 'Concerned'],
            'disgust': ['Disgusted', 'Disgust', 'Displeased'],
            'neutral': ['Neutral', 'Calm', 'Relaxed']
        }
        
        # Look for alternatives for this emotion
        alternatives = emotion_alternatives.get(emotion, [])
        
        for alt in alternatives:
            if alt in available_gestures:
                return alt
        
        # If no direct alternatives found, look for partial matches
        emotion_keywords = {
            'happy': ['smile', 'happy', 'joy', 'pleased'],
            'sad': ['sad', 'frown', 'down'],
            'angry': ['angry', 'mad', 'annoyed'],
            'surprise': ['surprise', 'shock', 'amaze'],
            'fear': ['worry', 'fear', 'scare', 'anxious'],
            'disgust': ['disgust', 'displease'],
            'neutral': ['neutral', 'calm', 'relax']
        }
        
        keywords = emotion_keywords.get(emotion, [])
        for keyword in keywords:
            for available_gesture in available_gestures:
                if keyword.lower() in available_gesture.lower():
                    return available_gesture
        
        return None
    
    def _initialize_face_detection(self):
        """Initialize face detection for robot camera"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("✅ Face detection initialized")
        except Exception as e:
            print(f"❌ Face detection initialization failed: {str(e)}")
            self.face_cascade = None
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition for robot microphone"""
        try:
            self.recognizer = sr.Recognizer()
            print("✅ Speech recognition initialized for Furhat")
        except Exception as e:
            print(f"⚠️  Speech recognition initialization warning: {str(e)}")
            self.recognizer = None
    
    def _initialize_camera(self):
        """Initialize camera capture for local device camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ Camera {self.camera_id} initialized successfully")
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {str(e)}")
            print("⚠️  FER functionality will be limited")
            self.cap = None
    
    def _load_fer_model(self, model_path: str):
        """Load the FER model"""
        print("🔄 Loading FER model...")
        
        if model_path is None:
            model_path = 'fer2013_final_model.pth'
        
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
                    './ter_distilbert_model',
                    './ter-model',
                    './models/ter_distilbert_model'
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
    
    def _preprocess_face(self, face_img):
        """Preprocess face image for FER"""
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        pil_img = Image.fromarray(face_img)
        tensor_img = self.fer_transform(pil_img)
        tensor_img = tensor_img.unsqueeze(0)
        
        return tensor_img.to(self.device)
    
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
    
    def _preprocess_face(self, face_img):
        """
        Preprocess face image for emotion recognition
        
        Args:
            face_img: OpenCV face image (grayscale or color)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(face_img)
        
        # Apply transforms
        tensor_img = self.fer_transform(pil_img)
        
        # Add batch dimension
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
    
    def _process_camera_fer(self, frame):
        """
        Process camera frame for facial expression recognition
        
        Args:
            frame: OpenCV camera frame
            
        Returns:
            tuple: (faces, predictions) where faces is list of (x,y,w,h) and predictions is list of (emotion, confidence)
        """
        if self.face_cascade is None or self.fer_model is None:
            return [], []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        predictions = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess face
            face_tensor = self._preprocess_face(face_roi)
            
            # Predict emotion
            emotion, confidence = self._predict_fer_emotion(face_tensor)
            predictions.append((emotion, confidence))
        
        return faces, predictions
    
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
        elif self.fusion_strategy == 'formula_based':
            return self._formula_based_fusion(fer_emotion, fer_confidence, ter_emotion, ter_confidence)
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
    
    def _formula_based_fusion(self, fer_emotion: str, fer_confidence: float, 
                             ter_emotion: str, ter_confidence: float) -> Tuple[str, float]:
        """
        Formula-based fusion strategy implementing the weighted combination formula:
        Pi = (∑i=1 to M ∑j=1 to N pi * modj * recij) / (∑i=1 to M ∑j=1 to N modj * recij)
        
        Where:
        - Pi = final prediction for emotion i
        - M = number of distinct emotions (7 in our case)
        - N = number of modalities (2: FER and TER)
        - pi = prediction confidence for emotion i
        - modj = accuracy of model j
        - recij = recall of emotion i for model j
        """
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
        
        # Calculate fusion scores for all emotions using the formula
        # Pi = (∑i=1 to M ∑j=1 to N pi * modj * recij) / (∑i=1 to M ∑j=1 to N modj * recij)
        emotion_scores = {}
        
        for emotion in self.emotion_labels:
            # Get prediction confidence for this emotion from each modality
            fer_pred_conf = fer_confidence if fer_emotion == emotion else 0.0
            ter_pred_conf = ter_confidence if ter_emotion == emotion else 0.0
            
            # Get model performance metrics for this specific emotion
            fer_accuracy = MODEL_PERFORMANCE['fer']['accuracy']
            ter_accuracy = MODEL_PERFORMANCE['ter']['accuracy']
            fer_recall_for_emotion = MODEL_PERFORMANCE['fer']['recall'].get(emotion, 0.5)
            ter_recall_for_emotion = MODEL_PERFORMANCE['ter']['recall'].get(emotion, 0.5)
            
            # Calculate weighted prediction using the corrected formula
            # Numerator: ∑j=1 to N (pi * modj * recij) for this emotion across all modalities
            numerator = (fer_pred_conf * fer_accuracy * fer_recall_for_emotion + 
                        ter_pred_conf * ter_accuracy * ter_recall_for_emotion)
            
            # Denominator: ∑j=1 to N (modj * recij) for this emotion across all modalities
            denominator = (fer_accuracy * fer_recall_for_emotion + ter_accuracy * ter_recall_for_emotion)
            
            # Calculate final prediction for this emotion
            if denominator > 0:
                emotion_scores[emotion] = numerator / denominator
            else:
                emotion_scores[emotion] = 0.0
        
        # Find the emotion with highest score
        if emotion_scores:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            best_confidence = emotion_scores[best_emotion]
            
            # The confidence is already normalized by the formula, just ensure it's within [0, 1]
            return best_emotion, min(best_confidence, 1.0)
        else:
            # Fallback to confidence-based fusion if formula fails
            return self._confidence_based_fusion(fer_emotion, fer_confidence, ter_emotion, ter_confidence)
    
    def _furhat_listen(self, timeout: float = 3.0) -> Optional[str]:
        """Use Furhat robot's microphone to listen for speech"""
        if self.furhat is None:
            return None
        
        try:
            # Use Furhat's listen method
            result = self.furhat.listen()
            
            if result and hasattr(result, 'message') and result.message:
                # Check if the result contains actual speech or timeout/error
                if result.message not in ['SILENCE', 'INTERRUPTED', 'FAILED']:
                    return result.message
            
            return None
                
        except Exception as e:
            print(f"Furhat listen error: {e}")
            return None
    
    def _robot_response_thread_func(self):
        """Background thread for robot emotional responses"""
        while self.running:
            try:
                if not self.prediction_queue.empty():
                    emotion_data = self.prediction_queue.get_nowait()
                    self._execute_robot_response(emotion_data)
                time.sleep(0.1)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Robot response error: {e}")
    
    def _execute_robot_response(self, emotion_data):
        """Execute robot emotional response"""
        if not self.enable_robot_responses or self.furhat is None:
            return
        
        current_time = time.time()
        if current_time - self.last_robot_response_time < self.robot_response_cooldown:
            return  # Cooldown period not elapsed
        
        emotion = emotion_data.get('emotion')
        confidence = emotion_data.get('confidence', 0.0)
        
        if emotion and confidence > CONFIDENCE_THRESHOLD:
            response_success = False
            
            try:
                # Perform gesture if available
                gesture = EMOTION_GESTURES.get(emotion)
                if gesture:
                    try:
                        # Use the Furhat Remote API gesture method according to spec
                        result = self.furhat.gesture(name=gesture, blocking=False)
                        if hasattr(result, 'success') and result.success:
                            print(f"🤖 Executed gesture: {gesture}")
                        else:
                            print(f"⚠️  Gesture '{gesture}' execution returned: {result}")
                    except Exception as gesture_error:
                        print(f"⚠️  Gesture '{gesture}' failed: {gesture_error}")
                        # Try a simple fallback - just log the error without another gesture attempt
                        print("🔧 Continuing without gesture...")
                
                # Say emotional response (this should work reliably)
                try:
                    responses = EMOTION_RESPONSES.get(emotion, ["I detected an emotion."])
                    response_text = np.random.choice(responses)
                    
                    # Use the Furhat Remote API say method according to spec
                    say_result = self.furhat.say(text=response_text, blocking=False)
                    if hasattr(say_result, 'success') and say_result.success:
                        print(f"🤖 Robot said: {response_text}")
                        response_success = True
                    else:
                        print(f"⚠️  Speech failed: {say_result}")
                        
                except Exception as speech_error:
                    print(f"⚠️  Speech failed: {speech_error}")
                
                # Update LED color based on emotion
                try:
                    color = self.emotion_colors.get(emotion, (128, 128, 128))
                    # Convert BGR to RGB for Furhat (API expects red, green, blue parameters)
                    r, g, b = color[2], color[1], color[0]
                    
                    led_result = self.furhat.set_led(red=r, green=g, blue=b)
                    if hasattr(led_result, 'success') and led_result.success:
                        print(f"🔵 LED color updated for {emotion}")
                    else:
                        print(f"⚠️  LED update result: {led_result}")
                        
                except Exception as led_error:
                    print(f"⚠️  LED update failed: {led_error}")
                
                # Always update cooldown if we attempted any response
                self.last_robot_response_time = current_time
                
                if response_success:
                    print(f"🤖 Robot response completed: {emotion} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"❌ Robot response execution error: {e}")
                # Still update the cooldown to prevent spam
                self.last_robot_response_time = current_time
    
    def _audio_processing_thread(self):
        """Background thread for continuous audio processing using Furhat"""
        while self.running:
            try:
                if self.voice_capture_active:
                    text = self._furhat_listen(timeout=2.0)
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
    
    def _get_camera_frame(self):
        """
        Get camera frame from local device camera
        Returns:
            numpy.ndarray or None: Camera frame if successful, None otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                return frame
            else:
                print("⚠️  Failed to capture frame from camera")
                return None
        except Exception as e:
            print(f"❌ Camera capture error: {e}")
            return None
    
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
        cv2.rectangle(frame, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 140), (255, 255, 255), 2)
        
        # FPS and device info
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        device_text = f"Device: {self.device.type.upper()}"
        cv2.putText(frame, device_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Furhat connection status
        furhat_status = "Connected" if self.furhat else "Disconnected"
        furhat_color = (0, 255, 0) if self.furhat else (0, 0, 255)
        cv2.putText(frame, f"Furhat: {furhat_status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, furhat_color, 1)
        
        # Current emotions
        fer_text = f"FER: {self.current_fer_emotion or 'None'}"
        ter_text = f"TER: {self.current_ter_emotion or 'None'}"
        fused_text = f"Fused: {self.current_fused_emotion or 'None'}"
        
        cv2.putText(frame, fer_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(frame, ter_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        cv2.putText(frame, fused_text, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
        
        # TER panel (bottom-left)
        if self.show_ter_panel:
            panel_y = height - 150
            cv2.rectangle(frame, (10, panel_y), (400, height - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, panel_y), (400, height - 10), (255, 255, 255), 2)
            
            # Voice capture status
            status_color = (0, 255, 0) if self.voice_capture_active else (0, 0, 255)
            status_text = "FURHAT LISTENING" if self.voice_capture_active else "PRESS 'V' TO ACTIVATE FURHAT MIC"
            cv2.putText(frame, status_text, (20, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, status_color, 2)
            
            # Recent TER prediction
            if hasattr(self, 'last_ter_text') and self.last_ter_text:
                text_display = self.last_ter_text[:35] + "..." if len(self.last_ter_text) > 35 else self.last_ter_text
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
            "Furhat Controls:",
            "Q - Quit",
            "V - Toggle Furhat Mic",
            "S - Save Frame",
            "T - Toggle TER Panel",
            "F - Fullscreen",
            "H - Show History",
            "P - Show Statistics",
            "R - Toggle Robot Responses",
            "G - Manual Gesture Test"
        ]
        
        for i, control in enumerate(controls):
            y_pos = 30 + (i * 18)
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, control, (width - 220, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, color, 1)
    
    def _save_frame(self, frame):
        """Save current frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"furhat_multimodal_emotion_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Frame saved as {filename}")
    
    def _show_history(self):
        """Display emotion prediction history"""
        if not self.prediction_history:
            print("📭 No prediction history available")
            return
        
        print(f"\n📚 Recent Furhat Multimodal Predictions (last 10):")
        print("-" * 90)
        
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
        
        print("\n📊 Furhat Session Statistics:")
        print(f"   Total Predictions: {total}")
        print("\n🎭 Emotion Distribution:")
        
        sorted_emotions = sorted(self.emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, count in sorted_emotions:
            percentage = (count / total) * 100
            bar_length = int(percentage / 5)
            bar = "█" * bar_length
            emoji = self.emotion_emojis.get(emotion, '❓')
            print(f"   {emotion:>8} {emoji}: {bar:<20} {count:2d} ({percentage:5.1f}%)")
    
    def _test_robot_gesture(self):
        """Test robot gesture manually"""
        if self.furhat is None:
            print("❌ Furhat not connected")
            return
        
        try:
            # Test a simple gesture first
            print("🧪 Testing robot gestures...")
            
            # Try BigSmile first as it's commonly available
            test_gestures = ['BigSmile', 'Smile', 'Nod', 'Surprised']
            
            for gesture_name in test_gestures:
                try:
                    result = self.furhat.gesture(name=gesture_name, blocking=False)
                    if hasattr(result, 'success') and result.success:
                        print(f"✅ Gesture '{gesture_name}' executed successfully")
                        self.furhat.say(text=f"Testing {gesture_name} gesture!", blocking=False)
                        break
                    else:
                        print(f"⚠️  Gesture '{gesture_name}' failed: {result}")
                except Exception as e:
                    print(f"⚠️  Gesture '{gesture_name}' error: {e}")
            else:
                print("❌ No test gestures worked")
                self.furhat.say(text="Gesture test completed, but no gestures worked!", blocking=False)
                
        except Exception as e:
            print(f"❌ Gesture test failed: {e}")
    
    def run(self):
        """Main Furhat multimodal emotion recognition loop"""
        if self.furhat is None:
            print("❌ Furhat robot is not connected")
            return
        
        print("🎯 Starting Furhat Multimodal Emotion Recognition...")
        print("🤖 Using Furhat robot's microphone + device camera for multimodal emotion detection")
        print("📷 Camera input will be used for facial expression recognition")
        print("🎤 Furhat microphone will be used for textual emotion recognition")
        print("Press 'V' to toggle Furhat voice capture, 'Q' to quit")
        
        # Start processing threads
        self.running = True
        
        # Audio processing thread for Furhat microphone
        if self.recognizer is not None:
            self.audio_thread = threading.Thread(target=self._audio_processing_thread, daemon=True)
            self.audio_thread.start()
        
        # Robot response thread
        if self.enable_robot_responses:
            self.robot_response_thread = threading.Thread(target=self._robot_response_thread_func, daemon=True)
            self.robot_response_thread.start()
        
        try:
            while True:
                # Get camera frame from device camera
                frame = self._get_camera_frame()
                
                if frame is None:
                    # Create a black frame as fallback if camera is not available
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera not available", 
                               (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # Process camera frame for FER
                faces, fer_predictions = self._process_camera_fer(frame)
                
                # Update FER results from camera
                if len(fer_predictions) > 0:
                    # Take the most confident prediction if multiple faces
                    best_prediction = max(fer_predictions, key=lambda x: x[1] if x[1] is not None else 0)
                    self.current_fer_emotion, self.fer_confidence = best_prediction
                else:
                    # Fallback to neutral with low confidence if no faces detected
                    self.current_fer_emotion = 'neutral'
                    self.fer_confidence = 0.1
                
                # Draw FER predictions on frame
                if len(faces) > 0 and len(fer_predictions) > 0:
                    self._draw_fer_predictions(frame, faces, fer_predictions)
                
                # Process audio queue from Furhat microphone
                self.current_ter_emotion = None
                self.ter_confidence = 0.0
                
                while not self.audio_queue.empty():
                    try:
                        audio_result = self.audio_queue.get_nowait()
                        self.current_ter_emotion = audio_result['emotion']
                        self.ter_confidence = audio_result['confidence']
                        self.last_ter_text = audio_result['text']
                        print(f"🎤 Furhat heard: '{audio_result['text']}' → {audio_result['emotion']} ({audio_result['confidence']:.3f})")
                        
                        # Queue for robot response
                        if self.enable_robot_responses:
                            self.prediction_queue.put({
                                'emotion': audio_result['emotion'],
                                'confidence': audio_result['confidence']
                            })
                        
                    except queue.Empty:
                        break
                
                # Fusion of FER (from camera) and TER (from Furhat microphone)
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
                
                # Draw UI on display frame
                self._draw_ui_panels(frame)
                self._update_fps()
                
                # Add Furhat-specific information overlay
                cv2.putText(frame, "FURHAT MULTIMODAL EMOTION RECOGNITION", 
                           (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, "Camera FER + Furhat TER | Speak to robot when active", 
                           (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, f"Robot IP: {self.furhat_ip} | Camera: {self.camera_id}", 
                           (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Display frame
                if self.fullscreen:
                    cv2.namedWindow('Furhat Multimodal Emotion Recognition', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('Furhat Multimodal Emotion Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.namedWindow('Furhat Multimodal Emotion Recognition', cv2.WINDOW_NORMAL)
                
                cv2.imshow('Furhat Multimodal Emotion Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self.voice_capture_active = not self.voice_capture_active
                    status = "ON" if self.voice_capture_active else "OFF"
                    print(f"🎤 Furhat voice capture: {status}")
                    if self.furhat:
                        response = "I'm listening now" if self.voice_capture_active else "Voice capture stopped"
                        self.furhat.say(text=response)
                elif key == ord('s'):
                    self._save_frame(frame)
                elif key == ord('t'):
                    self.show_ter_panel = not self.show_ter_panel
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    if not self.fullscreen:
                        cv2.setWindowProperty('Furhat Multimodal Emotion Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif key == ord('h'):
                    self._show_history()
                elif key == ord('p'):
                    self._show_statistics()
                elif key == ord('r'):
                    self.enable_robot_responses = not self.enable_robot_responses
                    status = "ENABLED" if self.enable_robot_responses else "DISABLED"
                    print(f"🤖 Robot responses: {status}")
                elif key == ord('g'):
                    self._test_robot_gesture()
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error during inference: {e}")
        finally:
            # Cleanup
            self.running = False
            if self.audio_thread:
                self.audio_thread.join(timeout=2.0)
            if self.robot_response_thread:
                self.robot_response_thread.join(timeout=2.0)
            if self.furhat:
                try:
                    self.furhat.say(text="Emotion recognition session ended. Goodbye!")
                    self.furhat.set_led(red=0, green=0, blue=0)  # Turn off LEDs
                except:
                    pass
            # Camera cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("🔄 Cleanup completed")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Furhat Multimodal Emotion Recognition System (FER + TER)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python furhat_multimodal_emotion_inference.py                                    # Default settings
  python furhat_multimodal_emotion_inference.py --fer_model ./my_fer_model.pth     # Custom FER model
  python furhat_multimodal_emotion_inference.py --ter_model ./my_ter_model         # Custom TER model
  python furhat_multimodal_emotion_inference.py --device cpu                       # Force CPU usage
  python furhat_multimodal_emotion_inference.py --furhat_ip 192.168.1.100         # Remote Furhat robot
  python furhat_multimodal_emotion_inference.py --camera_id 1                      # Use camera 1
  python furhat_multimodal_emotion_inference.py --fusion confidence_based          # Change fusion strategy
  python furhat_multimodal_emotion_inference.py --fusion formula_based             # Use formula-based fusion
  python furhat_multimodal_emotion_inference.py --no_robot_responses               # Disable robot responses
        """
    )
    
    parser.add_argument(
        '--fer_model', 
        type=str, 
        default=None,
        help='Path to FER model file (default: fer2013_final_model.pth)'
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
        '--furhat_ip', 
        type=str, 
        default=FURHAT_IP,
        help=f'IP address of Furhat robot (default: {FURHAT_IP})'
    )
    
    parser.add_argument(
        '--camera_id', 
        type=int, 
        default=0,
        help='Camera ID for device camera (default: 0)'
    )
    
    parser.add_argument(
        '--fusion', 
        choices=['weighted_average', 'confidence_based', 'formula_based'], 
        default='confidence_based',
        help='Fusion strategy for combining emotions (default: confidence_based)'
    )
    
    parser.add_argument(
        '--no_robot_responses', 
        action='store_true',
        help='Disable robot emotional responses (default: responses enabled)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🤖 Furhat Multimodal Emotion Recognition System")
    print("=" * 60)
    print("Combining FER (device camera) and TER (Furhat microphone)")
    print("=" * 60)
    
    try:
        # Initialize the Furhat multimodal system
        system = FurhatMultimodalEmotionInference(
            fer_model_path=args.fer_model,
            ter_model_path=args.ter_model,
            device=args.device,
            furhat_ip=args.furhat_ip,
            camera_id=args.camera_id,
            fusion_strategy=args.fusion,
            enable_robot_responses=not args.no_robot_responses
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
