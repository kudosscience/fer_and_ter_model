#!/usr/bin/env python3
"""
Test script to demonstrate the new formula-based fusion strategy
"""

import sys
import os

# Add the current directory to the path so we can import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main class
from src.furhat.furhat_multimodal_emotion_inference import FurhatMultimodalEmotionInference, MODEL_PERFORMANCE

def test_fusion_strategies():
    """Test and compare different fusion strategies"""
    
    print("🧪 Testing Formula-Based Fusion Strategy")
    print("=" * 50)
    
    # Create a mock instance (without actual model loading)
    class MockFusionTester:
        def __init__(self):
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            
        def _weighted_average_fusion(self, fer_emotion, fer_confidence, ter_emotion, ter_confidence):
            # Simplified version of weighted average
            if fer_emotion == ter_emotion:
                return fer_emotion, (fer_confidence * 0.6 + ter_confidence * 0.4) * 1.2
            elif fer_confidence * 0.6 > ter_confidence * 0.4:
                return fer_emotion, fer_confidence * 0.6 * 0.8
            else:
                return ter_emotion, ter_confidence * 0.4 * 0.8
        
        def _confidence_based_fusion(self, fer_emotion, fer_confidence, ter_emotion, ter_confidence):
            # Simplified version of confidence-based
            if fer_confidence > ter_confidence:
                boost = 1.1 if fer_emotion == ter_emotion else 0.9
                return fer_emotion, min(fer_confidence * boost, 1.0)
            else:
                boost = 1.1 if fer_emotion == ter_emotion else 0.9
                return ter_emotion, min(ter_confidence * boost, 1.0)
        
        def _formula_based_fusion(self, fer_emotion, fer_confidence, ter_emotion, ter_confidence):
            # Implementation of the formula-based fusion
            emotion_scores = {}
            
            for emotion in self.emotion_labels:
                fer_pred_conf = fer_confidence if fer_emotion == emotion else 0.0
                ter_pred_conf = ter_confidence if ter_emotion == emotion else 0.0
                
                fer_accuracy = MODEL_PERFORMANCE['fer']['accuracy']
                ter_accuracy = MODEL_PERFORMANCE['ter']['accuracy']
                fer_recall = MODEL_PERFORMANCE['fer']['recall'].get(emotion, 0.5)
                ter_recall = MODEL_PERFORMANCE['ter']['recall'].get(emotion, 0.5)
                
                numerator = (fer_pred_conf * fer_accuracy * fer_recall + 
                           ter_pred_conf * ter_accuracy * ter_recall)
                denominator = (fer_accuracy * fer_recall + ter_accuracy * ter_recall)
                
                if denominator > 0:
                    emotion_scores[emotion] = numerator / denominator
                else:
                    emotion_scores[emotion] = 0.0
            
            if emotion_scores:
                best_emotion = max(emotion_scores, key=emotion_scores.get)
                best_confidence = emotion_scores[best_emotion]
                return best_emotion, min(best_confidence, 1.0)
            else:
                return 'neutral', 0.0
    
    # Test scenarios
    test_cases = [
        # (fer_emotion, fer_confidence, ter_emotion, ter_confidence, description)
        ('happy', 0.85, 'happy', 0.90, "Both models agree on happy emotion"),
        ('angry', 0.75, 'sad', 0.80, "Models disagree: angry vs sad"),
        ('surprise', 0.60, 'neutral', 0.65, "Models disagree: surprise vs neutral"),
        ('fear', 0.95, 'fear', 0.70, "Both agree on fear, different confidences"),
        ('neutral', 0.50, 'happy', 0.85, "Low confidence neutral vs high confidence happy"),
    ]
    
    tester = MockFusionTester()
    
    print(f"Model Performance Metrics:")
    print(f"FER Accuracy: {MODEL_PERFORMANCE['fer']['accuracy']}")
    print(f"TER Accuracy: {MODEL_PERFORMANCE['ter']['accuracy']}")
    print()
    
    for i, (fer_emotion, fer_conf, ter_emotion, ter_conf, description) in enumerate(test_cases, 1):
        print(f"Test Case {i}: {description}")
        print(f"Input: FER='{fer_emotion}' (conf={fer_conf:.2f}), TER='{ter_emotion}' (conf={ter_conf:.2f})")
        
        # Test all three fusion strategies
        weighted_result = tester._weighted_average_fusion(fer_emotion, fer_conf, ter_emotion, ter_conf)
        confidence_result = tester._confidence_based_fusion(fer_emotion, fer_conf, ter_emotion, ter_conf)
        formula_result = tester._formula_based_fusion(fer_emotion, fer_conf, ter_emotion, ter_conf)
        
        print(f"Results:")
        print(f"  Weighted Average: {weighted_result[0]} (conf={weighted_result[1]:.3f})")
        print(f"  Confidence Based: {confidence_result[0]} (conf={confidence_result[1]:.3f})")
        print(f"  Formula Based:    {formula_result[0]} (conf={formula_result[1]:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    test_fusion_strategies()
