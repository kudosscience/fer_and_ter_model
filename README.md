# Facial and Textual Emotion Recognition System

A comprehensive emotion recognition system that combines facial emotion recognition (FER) and textual emotion recognition (TER) with multimodal fusion capabilities and Furhat robot integration.

## Project Structure

```text
fer_and_ter_model/
├── src/                          # Source code
│   ├── fer/                      # Facial Emotion Recognition
│   │   └── camera_fer_inference.py
│   ├── ter/                      # Textual Emotion Recognition  
│   │   ├── voice_ter_inference.py
│   │   └── setup_voice_ter.py
│   ├── multimodal/               # Multimodal Fusion
│   │   └── multimodal_emotion_inference.py
│   ├── furhat/                   # Furhat Robot Integration
│   │   └── furhat_multimodal_emotion_inference.py
│   └── utils/                    # Shared utilities
├── models/                       # Trained models
│   ├── fer2013_final_model.pth
│   └── ter_distilbert_model/
├── datasets/                     # Dataset files
│   └── multimodal_emotion_dataset.json
├── notebooks/                    # Jupyter notebooks
│   ├── fer2013_model_training.ipynb
│   ├── multimodal_emotion_fusion.ipynb
│   ├── multimodal_emotion_recognition.ipynb
│   └── textual_emotion_recognition_distilbert.ipynb
├── tests/                        # Test scripts
│   ├── test_corrected_formula.py
│   ├── test_fer_model.py
│   ├── test_formula_fusion.py
│   ├── test_furhat_integration.py
│   └── test_multimodal_system.py
├── demos/                        # Demo scripts
│   ├── demo_furhat_usage.py
│   ├── demo_multimodal_usage.py
│   ├── demo_usage.py
│   └── demo_voice_ter.py
├── docs/                         # Documentation
│   ├── DATASET_SETUP.md
│   ├── FER_PROJECT_SUMMARY.md
│   ├── FURHAT_INTEGRATION_SUMMARY.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── JUNIE.md
│   └── README_*.md files
├── requirements/                 # Requirements files
│   ├── requirements_backup.txt
│   ├── requirements_camera_inference.txt
│   ├── requirements_furhat.txt
│   ├── requirements_multimodal.txt
│   └── requirements_voice_ter.txt
└── README.md                     # This file
```

## Quick Start

1. **Install dependencies** for your specific use case:
   - For FER: `pip install -r requirements/requirements_camera_inference.txt`
   - For TER: `pip install -r requirements/requirements_voice_ter.txt`
   - For Multimodal: `pip install -r requirements/requirements_multimodal.txt`
   - For Furhat: `pip install -r requirements/requirements_furhat.txt`

2. **Run demos**:

   ```bash
   python demos/demo_usage.py           # Basic FER demo
   python demos/demo_voice_ter.py       # TER demo
   python demos/demo_multimodal_usage.py # Multimodal demo
   python demos/demo_furhat_usage.py    # Furhat integration demo
   ```

3. **Run tests**:

   ```bash
   python tests/test_fer_model.py
   python tests/test_multimodal_system.py
   ```

## Components

### FER (Facial Emotion Recognition)

- Real-time camera-based emotion detection
- CNN model trained on FER2013 dataset
- Supports 7 emotion classes

### TER (Textual Emotion Recognition)

- Voice-to-text emotion recognition
- DistilBERT-based model
- Supports multiple emotion categories

### Multimodal Fusion

- Combines FER and TER outputs
- Multiple fusion strategies available
- Improved accuracy through multi-modal approach

### Furhat Integration

- Social robot platform integration
- Real-time emotion feedback
- Interactive emotion recognition system

## Documentation

Detailed documentation for each component can be found in the `docs/` directory:

- `FER_PROJECT_SUMMARY.md` - FER system overview
- `FURHAT_INTEGRATION_SUMMARY.md` - Furhat integration details
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

## Development

The project uses a modular structure where each component is self-contained in its respective directory under `src/`. All modules are properly packaged with `__init__.py` files for easy importing.
