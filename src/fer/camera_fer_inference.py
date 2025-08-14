#!/usr/bin/env python3
"""
Real-time Facial Expression Recognition using Camera Input
Based on the FER2013 CNN model architecture from the Jupyter notebook.

This script captures video from the camera, detects faces, and predicts emotions
in real-time using the trained FER model.

Requirements:
- Trained model file: fer2013_final_model.pth
- Camera/webcam connected to the system
- Required packages: torch, cv2, numpy, PIL

Usage:
    python camera_fer_inference.py [--model_path MODEL_PATH] [--camera_id CAMERA_ID]

Controls:
    - Press 'q' to quit
    - Press 's' to save current frame with prediction
    - Press 'f' to toggle fullscreen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import os
import sys
from datetime import datetime


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


class FERInference:
    """Real-time Facial Expression Recognition inference class"""
    
    def __init__(self, model_path, device='auto', camera_id=0):
        """
        Initialize the FER inference system
        
        Args:
            model_path (str): Path to the trained model file
            device (str): Device to use ('auto', 'cpu', 'cuda')
            camera_id (int): Camera ID for cv2.VideoCapture
        """
        self.emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
        self.emotion_emojis = {
            0: '😠', 1: '🤢', 2: '😨', 3: '😊',
            4: '😢', 5: '😲', 6: '😐'
        }
        
        self.emotion_colors = {
            0: (0, 0, 255),    # Angry - Red
            1: (0, 128, 0),    # Disgust - Green
            2: (128, 0, 128),  # Fear - Purple
            3: (0, 255, 255),  # Happy - Yellow
            4: (255, 0, 0),    # Sad - Blue
            5: (0, 165, 255),  # Surprise - Orange
            6: (128, 128, 128) # Neutral - Gray
        }
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize camera
        self.camera_id = camera_id
        self.cap = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # UI state
        self.fullscreen = False
        
    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model instance
        model = EmotionCNN(num_classes=7, dropout_rate=0.5)
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint contains additional info (optimizer, metrics, etc.)
                state_dict = checkpoint['model_state_dict']
                print(f"✅ Loaded checkpoint with test accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
            else:
                # Checkpoint is just the state dict
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self.device)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        return model
    
    def _initialize_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Camera {self.camera_id} initialized successfully")
    
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
        tensor_img = self.transform(pil_img)
        
        # Add batch dimension
        tensor_img = tensor_img.unsqueeze(0)
        
        return tensor_img.to(self.device)
    
    def _predict_emotion(self, face_tensor):
        """
        Predict emotion from preprocessed face tensor
        
        Args:
            face_tensor: Preprocessed face tensor
            
        Returns:
            tuple: (predicted_class, confidence_scores)
        """
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence_scores = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence_scores
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _draw_predictions(self, frame, faces, predictions):
        """
        Draw bounding boxes and emotion predictions on frame
        
        Args:
            frame: OpenCV frame
            faces: List of face coordinates [(x, y, w, h), ...]
            predictions: List of (emotion_class, confidence_scores) tuples
        """
        for (x, y, w, h), (emotion_class, confidence_scores) in zip(faces, predictions):
            # Draw face bounding box
            color = self.emotion_colors[emotion_class]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare text
            emotion_name = self.emotion_labels[emotion_class]
            emoji = self.emotion_emojis[emotion_class]
            confidence = confidence_scores[emotion_class]
            
            # Draw emotion label
            label = f"{emotion_name} {emoji}"
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
            
            # Draw confidence bar
            bar_width = 100
            bar_height = 10
            bar_x = x
            bar_y = y + h + 10
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (64, 64, 64), -1)
            
            # Confidence bar
            conf_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                         color, -1)
    
    def _draw_ui(self, frame):
        """Draw UI elements like FPS, instructions etc."""
        height, width = frame.shape[:2]
        
        # Draw FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Draw device info
        device_text = f"Device: {self.device.type.upper()}"
        cv2.putText(frame, device_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "S - Save frame",
            "F - Toggle fullscreen"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - 80 + (i * 20)
            cv2.putText(frame, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
    
    def _save_frame(self, frame):
        """Save current frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fer_prediction_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
    
    def run(self):
        """Main inference loop"""
        try:
            self._initialize_camera()
            
            print("🎥 Starting real-time emotion recognition...")
            print("Press 'q' to quit, 's' to save frame, 'f' for fullscreen")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Process each detected face
                predictions = []
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Preprocess face
                    face_tensor = self._preprocess_face(face_roi)
                    
                    # Predict emotion
                    emotion_class, confidence_scores = self._predict_emotion(face_tensor)
                    predictions.append((emotion_class, confidence_scores))
                
                # Draw predictions
                if len(faces) > 0:
                    self._draw_predictions(frame, faces, predictions)
                
                # Draw UI elements
                self._draw_ui(frame)
                
                # Update FPS
                self._update_fps()
                
                # Display frame
                if self.fullscreen:
                    cv2.namedWindow('FER Real-time', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('FER Real-time', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.namedWindow('FER Real-time', cv2.WINDOW_NORMAL)
                
                cv2.imshow('FER Real-time', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(frame)
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    if not self.fullscreen:
                        cv2.setWindowProperty('FER Real-time', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error during inference: {e}")
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("🔄 Cleanup completed")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Real-time Facial Expression Recognition using camera input"
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='fer2013_final_model.pth',
        help='Path to the trained model file (default: fer2013_final_model.pth)'
    )
    parser.add_argument(
        '--camera_id', 
        type=int, 
        default=0,
        help='Camera ID for cv2.VideoCapture (default: 0)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for inference (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("Please ensure the trained model file is in the current directory")
        print("Expected file: fer2013_final_model.pth")
        sys.exit(1)
    
    # Create and run inference system
    try:
        fer_system = FERInference(
            model_path=args.model_path,
            device=args.device,
            camera_id=args.camera_id
        )
        fer_system.run()
    except Exception as e:
        print(f"❌ Failed to initialize FER system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
