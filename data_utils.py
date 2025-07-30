"""
Data Loading and Preparation Utilities
======================================

Helper functions for loading and preparing datasets for the multimodal emotion recognition system.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

# Emotion mapping (consistent with main model)
EMOTION_LABELS = ['joy', 'anger', 'disgust', 'sadness', 'fear', 'surprise']
EMOTION_ID_MAP = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}

class DatasetManager:
    """Manages dataset organization and loading for the multimodal emotion recognition system"""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.raw_path = self.data_root / "raw"
        self.processed_path = self.data_root / "processed"
        
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the current dataset"""
        stats = {
            'fer_images': self._count_fer_images(),
            'text_data': self._count_text_data(),
            'multimodal_data': self._count_multimodal_data()
        }
        return stats
    
    def _count_fer_images(self) -> Dict:
        """Count FER images by split and emotion"""
        fer_path = self.raw_path / "fer_images"
        counts = {}
        
        for split in ['train', 'test', 'validation']:
            split_path = fer_path / split
            if split_path.exists():
                counts[split] = {}
                for emotion in EMOTION_LABELS:
                    emotion_path = split_path / emotion
                    if emotion_path.exists():
                        counts[split][emotion] = len(list(emotion_path.glob('*')))
                    else:
                        counts[split][emotion] = 0
        
        return counts
    
    def _count_text_data(self) -> Dict:
        """Count text data by split"""
        text_path = self.raw_path / "text_data"
        counts = {}
        
        for split in ['train', 'test', 'validation']:
            json_file = text_path / f"{split}_text.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    counts[split] = len(data)
            else:
                counts[split] = 0
        
        return counts
    
    def _count_multimodal_data(self) -> Dict:
        """Count multimodal data by split"""
        multimodal_path = self.raw_path / "multimodal_data"
        counts = {}
        
        for split in ['train', 'test', 'validation']:
            json_file = multimodal_path / f"{split}_multimodal.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    counts[split] = len(data)
            else:
                counts[split] = 0
        
        return counts
    
    def validate_dataset(self) -> Dict[str, List[str]]:
        """Validate dataset integrity and return issues found"""
        issues = {
            'missing_files': [],
            'invalid_emotions': [],
            'broken_paths': [],
            'format_errors': []
        }
        
        # Check multimodal data
        multimodal_path = self.raw_path / "multimodal_data"
        for split in ['train', 'test', 'validation']:
            json_file = multimodal_path / f"{split}_multimodal.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    for item in data:
                        # Check required fields
                        required_fields = ['image_path', 'text', 'emotion', 'id']
                        for field in required_fields:
                            if field not in item:
                                issues['format_errors'].append(f"Missing {field} in {json_file}")
                        
                        # Check emotion validity
                        if 'emotion' in item and item['emotion'] not in EMOTION_LABELS:
                            issues['invalid_emotions'].append(f"Invalid emotion '{item['emotion']}' in {json_file}")
                        
                        # Check image path exists
                        if 'image_path' in item:
                            img_path = Path(item['image_path'])
                            if not img_path.exists():
                                issues['broken_paths'].append(f"Image not found: {img_path}")
                                
                except Exception as e:
                    issues['format_errors'].append(f"Error reading {json_file}: {str(e)}")
        
        return issues
    
    def create_sample_dataset(self, num_samples_per_emotion: int = 10):
        """Create a sample dataset for testing (generates placeholder data)"""
        print(f"Creating sample dataset with {num_samples_per_emotion} samples per emotion...")
        
        # Sample texts for each emotion
        sample_texts = {
            'joy': [
                "I'm feeling absolutely wonderful today!",
                "This is the best day ever!",
                "I'm so happy and excited!",
                "Everything is going perfectly!",
                "I feel amazing and full of energy!",
                "What a beautiful and joyful moment!",
                "I'm beaming with happiness!",
                "This brings me so much joy!",
                "I feel blessed and grateful!",
                "Pure happiness fills my heart!"
            ],
            'anger': [
                "This makes me incredibly angry!",
                "I'm furious about this situation!",
                "This is absolutely unacceptable!",
                "I'm so mad I could scream!",
                "This really gets on my nerves!",
                "I'm boiling with rage!",
                "This is completely outrageous!",
                "I'm fed up with this nonsense!",
                "This makes my blood boil!",
                "I'm absolutely livid!"
            ],
            'disgust': [
                "That's absolutely disgusting!",
                "This makes me feel sick!",
                "How revolting and gross!",
                "I find this utterly repulsive!",
                "That smell is nauseating!",
                "This is so gross and vile!",
                "I'm disgusted by this behavior!",
                "This makes me want to gag!",
                "How utterly repugnant!",
                "This is absolutely vile!"
            ],
            'sadness': [
                "I feel so sad and empty inside.",
                "This brings tears to my eyes.",
                "I'm feeling really down today.",
                "My heart feels heavy with sorrow.",
                "I feel completely hopeless.",
                "This makes me incredibly sad.",
                "I'm overwhelmed with sadness.",
                "I feel like crying right now.",
                "This fills me with deep sorrow.",
                "I'm feeling quite melancholy."
            ],
            'fear': [
                "I'm terrified and scared!",
                "This frightens me to my core!",
                "I'm shaking with fear!",
                "I'm afraid of what might happen!",
                "This gives me the chills!",
                "I'm paralyzed with terror!",
                "This scares me so much!",
                "I feel anxious and worried!",
                "I'm trembling with fear!",
                "This fills me with dread!"
            ],
            'surprise': [
                "Wow, that was totally unexpected!",
                "I can't believe this happened!",
                "What a shocking turn of events!",
                "This completely caught me off guard!",
                "I'm absolutely stunned!",
                "This is such a surprise!",
                "I never saw this coming!",
                "How incredibly unexpected!",
                "This blew my mind!",
                "What an amazing surprise!"
            ]
        }
        
        # Create multimodal dataset
        multimodal_data = []
        for emotion in EMOTION_LABELS:
            for i in range(num_samples_per_emotion):
                # Use cycling through available texts
                text = sample_texts[emotion][i % len(sample_texts[emotion])]
                
                sample = {
                    "image_path": f"data/raw/fer_images/train/{emotion}/sample_{emotion}_{i+1:03d}.jpg",
                    "text": text,
                    "emotion": emotion,
                    "id": f"sample_{emotion}_{i+1:03d}"
                }
                multimodal_data.append(sample)
        
        # Save to training file
        train_file = self.raw_path / "multimodal_data" / "train_multimodal.json"
        with open(train_file, 'w') as f:
            json.dump(multimodal_data, f, indent=2)
        
        print(f"Sample dataset created: {train_file}")
        print(f"Total samples: {len(multimodal_data)}")
        print(f"Samples per emotion: {num_samples_per_emotion}")
        
        return multimodal_data
    
    def print_dataset_summary(self):
        """Print a comprehensive summary of the dataset"""
        print("="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        stats = self.get_dataset_stats()
        
        # FER Images
        print("\n📸 FER Images:")
        print("-" * 40)
        fer_stats = stats['fer_images']
        for split in ['train', 'test', 'validation']:
            if split in fer_stats:
                total = sum(fer_stats[split].values())
                print(f"{split.capitalize():12s}: {total:4d} images")
                for emotion, count in fer_stats[split].items():
                    print(f"  {emotion:10s}: {count:4d}")
        
        # Text Data
        print("\n📝 Text Data:")
        print("-" * 40)
        text_stats = stats['text_data']
        for split in ['train', 'test', 'validation']:
            count = text_stats.get(split, 0)
            print(f"{split.capitalize():12s}: {count:4d} samples")
        
        # Multimodal Data
        print("\n🔄 Multimodal Data:")
        print("-" * 40)
        multimodal_stats = stats['multimodal_data']
        for split in ['train', 'test', 'validation']:
            count = multimodal_stats.get(split, 0)
            print(f"{split.capitalize():12s}: {count:4d} samples")
        
        # Validation
        print("\n✅ Dataset Validation:")
        print("-" * 40)
        issues = self.validate_dataset()
        total_issues = sum(len(v) for v in issues.values())
        if total_issues == 0:
            print("No issues found! ✅")
        else:
            print(f"Found {total_issues} issues:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"  {issue_type}: {len(issue_list)} issues")


def main():
    """Example usage of the DatasetManager"""
    manager = DatasetManager()
    
    # Print current dataset summary
    manager.print_dataset_summary()
    
    # Create sample dataset if no data exists
    stats = manager.get_dataset_stats()
    multimodal_total = sum(stats['multimodal_data'].values())
    
    if multimodal_total == 0:
        print("\nNo multimodal data found. Creating sample dataset...")
        manager.create_sample_dataset(num_samples_per_emotion=5)
        print("\nUpdated dataset summary:")
        manager.print_dataset_summary()


if __name__ == "__main__":
    main()
