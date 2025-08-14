#!/usr/bin/env python3
"""
Test script to demonstrate the corrected formula-based fusion strategy
"""

import sys
import os

# Add the current directory to the path so we can import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main class
from furhat_multimodal_emotion_inference import MODEL_PERFORMANCE

def test_corrected_formula_fusion():
    """Test the corrected formula-based fusion strategy"""
    
    print("🧪 Testing Corrected Formula-Based Fusion Strategy")
    print("=" * 60)
    print("Formula: Pi = (∑i=1 to M ∑j=1 to N pi * modj * recij) / (∑i=1 to M ∑j=1 to N modj * recij)")
    print("Where recij is recall of emotion i for model j")
    print("=" * 60)
    
    # Create a mock tester for the corrected formula
    class MockFormulaTester:
        def __init__(self):
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            
        def _corrected_formula_based_fusion(self, fer_emotion, fer_confidence, ter_emotion, ter_confidence):
            """Implementation of the corrected formula-based fusion"""
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
                return best_emotion, min(best_confidence, 1.0)
            else:
                return 'neutral', 0.0
    
    # Test scenario
    tester = MockFormulaTester()
    
    print(f"Model Performance Metrics:")
    print(f"FER Accuracy: {MODEL_PERFORMANCE['fer']['accuracy']}")
    print(f"TER Accuracy: {MODEL_PERFORMANCE['ter']['accuracy']}")
    print(f"FER Recall by emotion: {MODEL_PERFORMANCE['fer']['recall']}")
    print(f"TER Recall by emotion: {MODEL_PERFORMANCE['ter']['recall']}")
    print()
    
    # Test case: Models disagree on emotions
    fer_emotion, fer_conf = 'happy', 0.85
    ter_emotion, ter_conf = 'sad', 0.75
    
    print(f"Test Case: Models disagree")
    print(f"Input: FER='{fer_emotion}' (conf={fer_conf:.2f}), TER='{ter_emotion}' (conf={ter_conf:.2f})")
    
    result = tester._corrected_formula_based_fusion(fer_emotion, fer_conf, ter_emotion, ter_conf)
    
    print(f"Result: {result[0]} (confidence={result[1]:.3f})")
    print()
    
    # Show the calculation details for each emotion
    print("Detailed calculation for each emotion:")
    print("-" * 40)
    
    for emotion in tester.emotion_labels:
        fer_pred_conf = fer_conf if fer_emotion == emotion else 0.0
        ter_pred_conf = ter_conf if ter_emotion == emotion else 0.0
        
        fer_accuracy = MODEL_PERFORMANCE['fer']['accuracy']
        ter_accuracy = MODEL_PERFORMANCE['ter']['accuracy']
        fer_recall = MODEL_PERFORMANCE['fer']['recall'].get(emotion, 0.5)
        ter_recall = MODEL_PERFORMANCE['ter']['recall'].get(emotion, 0.5)
        
        numerator = (fer_pred_conf * fer_accuracy * fer_recall + 
                    ter_pred_conf * ter_accuracy * ter_recall)
        denominator = (fer_accuracy * fer_recall + ter_accuracy * ter_recall)
        
        score = numerator / denominator if denominator > 0 else 0.0
        
        print(f"{emotion:>8}: Pi = {score:.3f} (FER_contrib={fer_pred_conf * fer_accuracy * fer_recall:.3f}, TER_contrib={ter_pred_conf * ter_accuracy * ter_recall:.3f})")

if __name__ == "__main__":
    test_corrected_formula_fusion()
