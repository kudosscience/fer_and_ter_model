# Dataset Directory Structure

This directory contains all data for the multimodal emotion recognition system.

## Directory Structure

```
data/
├── raw/                          # Raw, unprocessed data
│   ├── fer_images/              # Facial emotion recognition images
│   │   ├── train/               # Training images
│   │   │   ├── joy/             # Joy emotion images
│   │   │   ├── anger/           # Anger emotion images
│   │   │   ├── disgust/         # Disgust emotion images
│   │   │   ├── sadness/         # Sadness emotion images
│   │   │   ├── fear/            # Fear emotion images
│   │   │   └── surprise/        # Surprise emotion images
│   │   ├── test/                # Test images (same structure as train)
│   │   │   ├── joy/
│   │   │   ├── anger/
│   │   │   ├── disgust/
│   │   │   ├── sadness/
│   │   │   ├── fear/
│   │   │   └── surprise/
│   │   └── validation/          # Validation images (same structure as train)
│   │       ├── joy/
│   │       ├── anger/
│   │       ├── disgust/
│   │       ├── sadness/
│   │       ├── fear/
│   │       └── surprise/
│   ├── text_data/               # Text emotion recognition data
│   │   ├── train_text.json      # Training text data with labels
│   │   ├── test_text.json       # Test text data with labels
│   │   └── validation_text.json # Validation text data with labels
│   └── multimodal_data/         # Combined image + text data
│       ├── train_multimodal.json    # Training multimodal data
│       ├── test_multimodal.json     # Test multimodal data
│       └── validation_multimodal.json # Validation multimodal data
└── processed/                   # Processed/preprocessed data
    ├── fer_features/            # Extracted FER features
    │   ├── train_features.npy
    │   ├── test_features.npy
    │   └── validation_features.npy
    └── ter_features/            # Extracted TER features
        ├── train_features.npy
        ├── test_features.npy
        └── validation_features.npy
```

## Data Format Guidelines

### FER Images
- **Format**: JPG, PNG, or other image formats
- **Size**: Recommended 48x48 pixels (will be resized automatically)
- **Color**: Grayscale or RGB (will be converted to grayscale)
- **Naming**: Use descriptive names like `person1_joy_001.jpg`

### Text Data JSON Format
```json
[
  {
    "text": "I'm feeling absolutely wonderful today!",
    "emotion": "joy",
    "id": "text_001"
  },
  {
    "text": "This situation makes me really angry.",
    "emotion": "anger", 
    "id": "text_002"
  }
]
```

### Multimodal Data JSON Format
```json
[
  {
    "image_path": "data/raw/fer_images/train/joy/person1_joy_001.jpg",
    "text": "I'm feeling absolutely wonderful today!",
    "emotion": "joy",
    "id": "multimodal_001"
  },
  {
    "image_path": "data/raw/fer_images/train/anger/person2_anger_001.jpg",
    "text": "This situation makes me really angry.",
    "emotion": "anger",
    "id": "multimodal_002"
  }
]
```

## Emotion Classes
The system supports 6 emotion classes:
1. **joy** - Happy, positive emotions
2. **anger** - Angry, frustrated emotions
3. **disgust** - Disgusted, repulsed emotions
4. **sadness** - Sad, depressed emotions
5. **fear** - Fearful, anxious emotions
6. **surprise** - Surprised, shocked emotions

## Dataset Requirements

### For Training
- **Minimum per class**: 100 samples per emotion for meaningful training
- **Recommended per class**: 500+ samples per emotion for robust performance
- **Balance**: Try to maintain roughly equal samples across all emotion classes

### Data Splits
- **Training**: 70-80% of total data
- **Validation**: 10-15% of total data (for hyperparameter tuning)
- **Test**: 10-15% of total data (for final evaluation)

## Popular Datasets You Can Use

### For FER (Facial Emotion Recognition)
1. **FER2013**: Classic facial emotion dataset
2. **AffectNet**: Large-scale facial expression dataset
3. **RAF-DB**: Real-world Affective Face Database
4. **KDEF**: Karolinska Directed Emotional Faces

### For TER (Text Emotion Recognition)
1. **GoEmotions**: Google's emotion dataset
2. **SemEval-2018 Task 1**: Emotion classification dataset
3. **ISEAR**: International Survey on Emotion Antecedents and Reactions
4. **Emotion Dataset**: Various Twitter/social media emotion datasets

### For Multimodal
1. **CMU-MOSEI**: Multimodal sentiment and emotion analysis
2. **IEMOCAP**: Interactive Emotional Dyadic Motion Capture
3. **MELD**: Multimodal EmotionLines Dataset

## Usage Notes

1. Place your image files in the appropriate emotion subdirectories
2. Create JSON files for text and multimodal data following the specified format
3. Ensure image paths in multimodal JSON files are relative to the project root
4. The system will automatically preprocess images (resize, normalize, etc.)
5. Text will be tokenized and processed using DistilBERT tokenizer

## Data Preprocessing

The system will automatically:
- Resize images to 48x48 pixels
- Convert images to grayscale
- Normalize pixel values to [0, 1]
- Tokenize text using DistilBERT tokenizer
- Pad/truncate text to max_length (128 tokens)
- Convert emotion labels to numerical indices
