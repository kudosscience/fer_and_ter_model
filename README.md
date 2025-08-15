# Multimodal Emotion Recognition System

A comprehensive, real-time emotion recognition system that combines **Facial Emotion Recognition (FER)** and **Textual Emotion Recognition (TER)** with advanced multimodal fusion capabilities and **Furhat robot integration**. This system provides accurate emotion detection through computer vision and natural language processing, with support for interactive robotics applications.

## 🌟 Key Features

- **🎭 Facial Emotion Recognition**: Real-time emotion detection from camera feed using CNN models trained on FER2013
- **💬 Textual Emotion Recognition**: Voice-to-text emotion analysis using DistilBERT-based models  
- **🔀 Multimodal Fusion**: Advanced fusion strategies (confidence-based, weighted average, and formula-based)
- **🤖 Furhat Integration**: Complete social robot platform integration with real-time emotion feedback
- **⚡ Real-time Processing**: Live emotion recognition with interactive GUI interfaces
- **🛡️ Robust Architecture**: Comprehensive error handling and fallback mechanisms
- **📊 Comprehensive Testing**: Full test suite with system health checks and validation

## 🧪 Testing & Validation

The system includes comprehensive testing capabilities to ensure reliability:

### Run All Tests

```bash
# Run comprehensive system tests
python tests/test_multimodal_system.py    # Complete system validation
python tests/test_fer_model.py           # FER component testing
python tests/test_furhat_integration.py  # Furhat integration testing
python tests/test_formula_fusion.py      # Fusion algorithm testing
python tests/test_corrected_formula.py   # Mathematical validation
```

### System Health Check

```bash
# Validate system configuration and dependencies
python demos/demo_multimodal_usage.py
```

The test suite validates:

- Model loading and initialization
- Camera and microphone availability  
- Emotion fusion algorithm accuracy
- Robot integration functionality
- Error handling and fallback mechanisms

## 📁 Project Architecture

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

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (3.11+ recommended)
- Webcam for facial emotion recognition
- Microphone for voice/text emotion recognition
- GPU support recommended for optimal performance

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kudosscience/fer_and_ter_model.git
   cd fer_and_ter_model
   ```

2. **Install the package**:

   ```bash
   # Install with all components
   pip install -e ".[all]"
   
   # OR install specific components:
   pip install -e ".[multimodal]"    # For multimodal processing
   pip install -e ".[furhat]"        # For Furhat robot integration
   pip install -e ".[camera_inference]"  # For FER only
   pip install -e ".[voice_ter]"     # For TER only
   ```

3. **Alternative: Install from requirements files**:

   ```bash
   # Choose based on your use case:
   pip install -r requirements/requirements_multimodal.txt     # Recommended
   pip install -r requirements/requirements_furhat.txt         # For robot integration
   pip install -r requirements/requirements_camera_inference.txt  # FER only
   pip install -r requirements/requirements_voice_ter.txt      # TER only
   ```

### Quick Demo

```bash
# Run the comprehensive multimodal demo
python demos/demo_multimodal_usage.py

# Try individual components
python demos/demo_usage.py           # Basic FER demo
python demos/demo_voice_ter.py       # TER demo  
python demos/demo_furhat_usage.py    # Furhat integration demo
```

### Console Commands

After installation, you can use these console commands:

```bash
# Run individual components
fer-camera              # Launch facial emotion recognition
ter-voice              # Launch textual emotion recognition  
multimodal-emotion     # Launch multimodal system
furhat-emotion         # Launch Furhat integration
```

## 🧩 System Components

### 🎭 FER (Facial Emotion Recognition)

- **Real-time Processing**: Live camera feed emotion detection with GUI overlay
- **CNN Architecture**: Deep learning model trained on FER2013 dataset  
- **7 Emotion Classes**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- **High Accuracy**: Optimized model with robust preprocessing pipeline
- **Fallback Support**: Graceful handling of camera unavailability

**Usage**:

```bash
# Direct script execution
python src/fer/camera_fer_inference.py

# Console command
fer-camera
```

### 💬 TER (Textual Emotion Recognition)

- **Voice-to-Text Pipeline**: Real-time speech recognition and emotion analysis
- **DistilBERT Model**: State-of-the-art transformer-based emotion classification
- **Multi-emotion Support**: Comprehensive emotion category detection
- **Robust Processing**: Advanced text preprocessing and normalization
- **Audio Fallback**: Multiple audio input handling strategies

**Usage**:

```bash
# Direct script execution  
python src/ter/voice_ter_inference.py

# Console command
ter-voice
```

### 🔀 Multimodal Fusion System

The core innovation of this system - combines FER and TER for superior accuracy:

- **Multiple Fusion Strategies**:
  - **Confidence-based**: Selects prediction with highest confidence score
  - **Weighted Average**: Combines predictions with 60% facial, 40% textual weighting
  - **Formula-based**: Advanced mathematical fusion using custom algorithm

- **Real-time Integration**: Simultaneous processing of visual and audio streams
- **Adaptive Fallback**: Works with single modality when needed
- **Interactive GUI**: Live visualization of both streams and fusion results

**Usage**:

```bash
# Default multimodal processing
python src/multimodal/multimodal_emotion_inference.py

# With specific fusion strategy
python src/multimodal/multimodal_emotion_inference.py --fusion confidence_based
python src/multimodal/multimodal_emotion_inference.py --fusion weighted_average  
python src/multimodal/multimodal_emotion_inference.py --fusion formula_based

# Console command
multimodal-emotion
```

### 🤖 Furhat Robot Integration

Complete social robotics platform integration for interactive emotion recognition:

- **Furhat Remote API**: Official SDK integration for robot communication
- **Interactive Responses**: Robot gestures, speech, and LED feedback based on detected emotions
- **Voice Integration**: Uses robot's microphone for natural interaction
- **Real-time Feedback**: Immediate emotional responses and social cues
- **Robust Connection**: Graceful fallback when robot unavailable

**Robot Capabilities**:

- Emotional gesture mapping (BigSmile, Frown, Surprised, etc.)
- LED color changes reflecting emotional states  
- Speech synthesis for emotion acknowledgment
- Interactive conversation flow

**Usage**:

```bash
# Furhat integration (requires robot connection)
python src/furhat/furhat_multimodal_emotion_inference.py

# With fusion strategy
python src/furhat/furhat_multimodal_emotion_inference.py --fusion formula_based

# Console command  
furhat-emotion
```

## 📚 Comprehensive Documentation

Detailed documentation for each component is available in the `docs/` directory:

### Core Documentation

- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** - Complete technical implementation overview
- **[FER_PROJECT_SUMMARY.md](docs/FER_PROJECT_SUMMARY.md)** - Facial emotion recognition deep dive  
- **[FURHAT_INTEGRATION_SUMMARY.md](docs/FURHAT_INTEGRATION_SUMMARY.md)** - Robot integration guide
- **[DATASET_SETUP.md](docs/DATASET_SETUP.md)** - Dataset preparation and training

### Component-Specific Guides

- **[README_multimodal.md](docs/README_multimodal.md)** - Multimodal system usage and configuration
- **[README_furhat.md](docs/README_furhat.md)** - Furhat robot setup and integration
- **[README_camera_inference.md](docs/README_camera_inference.md)** - FER system configuration
- **[README_voice_ter.md](docs/README_voice_ter.md)** - Voice/text emotion recognition setup

## ⚙️ Advanced Configuration

### Custom Model Usage

```bash
# Use custom FER model
python src/multimodal/multimodal_emotion_inference.py \
    --fer_model ./path/to/custom_fer_model.pth

# Use custom TER model  
python src/multimodal/multimodal_emotion_inference.py \
    --ter_model ./path/to/custom_ter_model/
```

### Fusion Strategy Selection

```bash
# Confidence-based fusion (chooses most confident prediction)
python src/multimodal/multimodal_emotion_inference.py --fusion confidence_based

# Weighted average fusion (60% facial, 40% textual)  
python src/multimodal/multimodal_emotion_inference.py --fusion weighted_average

# Formula-based fusion (mathematical optimization)
python src/multimodal/multimodal_emotion_inference.py --fusion formula_based
```

### Performance Optimization

```bash
# GPU acceleration (if available)
export CUDA_VISIBLE_DEVICES=0
python src/multimodal/multimodal_emotion_inference.py

# CPU-only mode
export CUDA_VISIBLE_DEVICES=""
python src/multimodal/multimodal_emotion_inference.py
```

## 🔧 Development & Extension

### Project Structure Overview

The system follows a modular architecture allowing easy extension and modification:

- **`src/`** - Core source code with modular components
- **`models/`** - Pre-trained models (FER CNN, TER DistilBERT)
- **`datasets/`** - Training and evaluation datasets
- **`tests/`** - Comprehensive test suite
- **`demos/`** - Usage examples and demonstrations
- **`docs/`** - Detailed technical documentation
- **`requirements/`** - Component-specific dependency management

### Adding New Components

1. Create module in appropriate `src/` subdirectory
2. Add requirements to relevant requirements file
3. Create tests in `tests/` directory
4. Add demo in `demos/` directory
5. Update documentation in `docs/`

### Contributing

This project uses professional development practices:

- **Modular Design**: Each component is self-contained and reusable
- **Comprehensive Testing**: Full test coverage with validation scripts
- **Documentation**: Extensive documentation for all components
- **Package Management**: Proper Python packaging with setuptools
- **Console Integration**: Command-line tools for easy usage

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected**:

```bash
# Test camera access
python -c "import cv2; print('Camera available:', cv2.VideoCapture(0).isOpened())"
```

**Microphone issues**:

```bash
# Test microphone access  
python -c "import speech_recognition as sr; print('Microphone available:', len(sr.Microphone.list_microphone_names()) > 0)"
```

**Model loading errors**:

- Ensure models are downloaded and in `models/` directory
- Check file permissions and paths
- Verify CUDA availability for GPU models

**Furhat connection issues**:

- Verify robot IP address and port
- Check network connectivity
- Ensure Furhat Remote API service is running

### Performance Tips

- Use GPU acceleration when available
- Close other applications using camera/microphone
- Ensure adequate lighting for facial recognition
- Use external microphone for better voice recognition
- Run system health check before important usage

## 📊 Model Information

### FER Model (Facial Emotion Recognition)

- **Architecture**: Custom CNN trained on FER2013
- **Input**: 48x48 grayscale facial images
- **Output**: 7 emotion classes (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **Accuracy**: Optimized for real-time performance with robust preprocessing

### TER Model (Textual Emotion Recognition)

- **Architecture**: DistilBERT-based transformer
- **Input**: Text transcribed from speech
- **Output**: Multi-dimensional emotion classification
- **Features**: Advanced text preprocessing and normalization

### Fusion Algorithms

- **Confidence-based**: Selects highest confidence prediction
- **Weighted Average**: Optimized 60/40 facial/textual weighting
- **Formula-based**: Mathematical fusion using correlation analysis

## 🎯 Use Cases

- **Research**: Emotion recognition research and experimentation
- **Education**: Teaching multimodal AI and emotion recognition
- **Healthcare**: Patient emotion monitoring and therapy assistance  
- **Human-Computer Interaction**: Emotional interfaces and feedback systems
- **Social Robotics**: Interactive robots with emotional intelligence
- **Accessibility**: Emotion-aware assistive technologies

## 📈 Performance Metrics

The system has been validated across multiple scenarios:

- **Real-time Processing**: < 100ms latency for combined FER+TER
- **Accuracy**: Improved performance through multimodal fusion
- **Robustness**: Graceful degradation with single modality
- **Scalability**: Modular architecture supports easy extension

## 🔗 Related Projects

- [FER2013 Dataset](https://www.kaggle.com/deadskull7/fer2013)
- [Furhat Robot Platform](https://furhatrobotics.com/)
- [DistilBERT Model](https://huggingface.co/distilbert-base-uncased)
- [OpenCV Computer Vision](https://opencv.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Henry Ward

- GitHub: [@kudosscience](https://github.com/kudosscience)
- Email: [45144290+kudosscience@users.noreply.github.com](mailto:45144290+kudosscience@users.noreply.github.com)

## 🙏 Acknowledgments

- FER2013 dataset contributors
- Hugging Face Transformers library
- OpenCV community
- Furhat Robotics platform
- PyTorch and scikit-learn teams

## 📞 Support

For issues, questions, or contributions:

1. Check the [Issues](https://github.com/kudosscience/fer_and_ter_model/issues) page
2. Review the comprehensive documentation in `docs/`
3. Run the system health check: `python demos/demo_multimodal_usage.py`
4. Create a new issue with detailed information

---

Built with ❤️ for multimodal emotion recognition research and applications
