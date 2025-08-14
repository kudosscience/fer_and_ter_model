"""
Emotion Recognition System

This package contains modules for facial emotion recognition (FER),
textual emotion recognition (TER), multimodal emotion fusion,
and Furhat robot integration.
"""

__version__ = "1.0.0"
__author__ = "Henry Ward"

# Import main classes for easy access
try:
    from .fer.camera_fer_inference import EmotionCNN, CameraFERInference
except ImportError:
    pass

try:
    from .ter.voice_ter_inference import VoiceTERInference
except ImportError:
    pass

try:
    from .multimodal.multimodal_emotion_inference import MultimodalEmotionInference
except ImportError:
    pass

try:
    from .furhat.furhat_multimodal_emotion_inference import FurhatMultimodalEmotionInference
except ImportError:
    pass

__all__ = [
    'EmotionCNN',
    'CameraFERInference',
    'VoiceTERInference', 
    'MultimodalEmotionInference',
    'FurhatMultimodalEmotionInference'
]
