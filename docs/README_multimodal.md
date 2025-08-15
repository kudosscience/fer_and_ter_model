# Multimodal Emotion Recognition System

A real-time emotion recognition system that combines **Facial Expression Recognition (FER)** and **Textual Emotion Recognition (TER)** using camera and microphone input.

## Overview

This system simultaneously processes:
- **Visual emotions** from facial expressions captured via camera
- **Textual emotions** from speech transcribed via microphone
- **Fused emotions** combining both modalities for improved accuracy

## Features

### 🎭 Multimodal Processing
- Real-time facial expression recognition using CNN model
- Voice-to-text transcription with emotion analysis using DistilBERT
- Advanced fusion strategies for combining emotions
- Confidence-based decision making

### 🎮 Interactive Interface
- Live video feed with emotion overlays
- Real-time voice capture with visual feedback
- Configurable UI panels and displays
- Keyboard controls for all functions

### 📊 Analytics & Tracking
- Session statistics and emotion distribution
- Prediction history and confidence tracking
- Performance monitoring (FPS, accuracy)
- Export capabilities for data analysis

### ⚙️ Flexible Configuration
- Multiple fusion strategies (weighted average, confidence-based)
- Customizable model paths and parameters
- Device selection (CPU/CUDA)
- Adjustable confidence thresholds

## Installation

### 1. Clone or Download
```bash
# If using git
git clone <repository-url>
cd fer_and_ter_model

# Or download and extract the project files
```

### 2. Install Dependencies
```bash
# Install all requirements
pip install -r requirements_multimodal.txt

# Or install individual components:
pip install torch torchvision transformers scikit-learn
pip install opencv-python pillow speechrecognition pyaudio
pip install numpy pandas tqdm
```

### 3. Platform-Specific Setup

#### Windows
```bash
# If pyaudio installation fails:
conda install pyaudio
# Or download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

#### Linux
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install libgl1-mesa-glx  # For OpenCV
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev
```

#### macOS
```bash
brew install portaudio
pip install pyaudio
```

## Required Models

### FER Model
- **File**: `fer2013_final_model.pth`
- **Description**: Trained CNN model for facial expression recognition
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### TER Model
- **Directory**: `ter_distilbert_model/`
- **Description**: Fine-tuned DistilBERT model for text emotion recognition
- **Files needed**:
  - `config.json`
  - `model.safetensors`
  - `tokenizer_config.json`
  - `vocab.txt`
  - `label_encoder.pkl`

## Usage

### Basic Usage
```bash
# Run with default settings
python multimodal_emotion_inference.py

# Use specific models
python multimodal_emotion_inference.py --fer_model ./models/my_fer_model.pth --ter_model ./models/my_ter_model

# Force CPU usage
python multimodal_emotion_inference.py --device cpu

# Use different camera
python multimodal_emotion_inference.py --camera_id 1

# Change fusion strategy
python multimodal_emotion_inference.py --fusion weighted_average
```

### Interactive Controls

Once running, use these keyboard controls:

| Key | Function |
|-----|----------|
| `V` | Toggle voice capture ON/OFF |
| `Q` | Quit the application |
| `S` | Save current frame with predictions |
| `T` | Toggle TER panel visibility |
| `F` | Toggle fullscreen mode |
| `H` | Show prediction history |
| `P` | Print session statistics |

## How It Works

### 1. Facial Expression Recognition (FER)
- Captures video frames from camera
- Detects faces using Haar cascades
- Preprocesses face regions (48x48 grayscale)
- Predicts emotions using trained CNN
- Displays bounding boxes with emotion labels

### 2. Textual Emotion Recognition (TER)
- Continuously listens to microphone (when activated)
- Transcribes speech using Google Speech Recognition
- Cleans and preprocesses text
- Predicts emotions using fine-tuned DistilBERT
- Shows transcribed text and confidence scores

### 3. Multimodal Fusion
Two fusion strategies are available:

#### Confidence-Based Fusion (Default)
- Selects the emotion with higher confidence
- Boosts confidence when both modalities agree
- Reduces confidence when modalities disagree

#### Weighted Average Fusion
- Combines emotions using predefined weights:
  - Facial: 60%
  - Textual: 40%
- Adjusts confidence based on agreement

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Camera Input  │    │ Microphone Input│
│                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  Face Detection │    │ Speech-to-Text  │
│   & FER Model   │    │   & TER Model   │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ FER Predictions │    │ TER Predictions │
│   + Confidence  │    │   + Confidence  │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └─────────┬────────────┘
                    ▼
          ┌─────────────────┐
          │ Emotion Fusion  │
          │   Algorithm     │
          └─────────┬───────┘
                    ▼
          ┌─────────────────┐
          │ Final Emotion   │
          │  + Confidence   │
          └─────────────────┘
```

## Configuration Options

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fer_model` | str | `fer2013_final_model.pth` | Path to FER model file |
| `--ter_model` | str | Auto-detect | Path to TER model directory |
| `--device` | str | `auto` | Device: auto, cpu, cuda |
| `--camera_id` | int | `0` | Camera ID for video capture |
| `--fusion` | str | `confidence_based` | Fusion strategy |

### Fusion Strategies

#### 1. Confidence-Based (Recommended)
- Chooses emotion with highest confidence
- Considers both modalities equally
- Ideal for balanced scenarios

#### 2. Weighted Average
- Uses fixed weights (60% facial, 40% textual)
- Better for scenarios where one modality is consistently more reliable
- Customizable weights in code

### Adjustable Parameters

Edit these constants in the code for fine-tuning:

```python
# Fusion weights (for weighted_average strategy)
FUSION_WEIGHTS = {
    'facial': 0.6,    # Weight for facial emotion
    'textual': 0.4    # Weight for textual emotion
}

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.3

# Update interval for processing
UPDATE_INTERVAL = 0.1  # seconds
```

## Performance Optimization

### For Better Performance:
1. **Use CUDA**: Install PyTorch with CUDA support
2. **Adjust resolution**: Modify camera resolution in code
3. **Optimize models**: Use model quantization or pruning
4. **Reduce update frequency**: Increase `UPDATE_INTERVAL`

### Typical Performance:
- **FPS**: 15-30 (depending on hardware)
- **Latency**: 100-300ms per prediction
- **Memory**: 1-3GB (depending on models and device)

## Troubleshooting

### Common Issues

#### 1. Camera Not Working
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Try different camera ID
python multimodal_emotion_inference.py --camera_id 1
```

#### 2. Microphone Not Working
```bash
# Test microphone
python -c "import speech_recognition as sr; r = sr.Recognizer(); print('Microphone test:', r.recognize_google(r.listen(sr.Microphone())))"

# Check available microphones
python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
```

#### 3. PyAudio Installation Issues
```bash
# Windows
conda install pyaudio

# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio
```

#### 4. Model Loading Errors
- Ensure model files are in the correct locations
- Check file permissions
- Verify model file integrity
- Use absolute paths if relative paths fail

#### 5. CUDA Out of Memory
```bash
# Force CPU usage
python multimodal_emotion_inference.py --device cpu

# Or reduce batch sizes in the code
```

### Debug Mode
Add debug prints by modifying the code or run with Python's verbose mode:
```bash
python -v multimodal_emotion_inference.py
```

## Data Export

### Saving Predictions
- **Frames**: Press `S` to save annotated video frames
- **History**: Access via `_show_history()` method
- **Statistics**: Use `_show_statistics()` method

### Custom Export
Modify the code to add JSON/CSV export functionality:
```python
# In the prediction loop, add:
with open('predictions.json', 'a') as f:
    json.dump(prediction_result, f)
    f.write('\n')
```

## Extension Ideas

### Potential Enhancements:
1. **Audio Emotion Recognition**: Add direct audio emotion recognition
2. **Real-time Streaming**: WebRTC or RTMP streaming
3. **Database Integration**: Store predictions in database
4. **REST API**: Create API endpoints for remote access
5. **Multiple Faces**: Handle multiple faces simultaneously
6. **Emotion Trends**: Track emotion changes over time
7. **Custom Models**: Support for different model architectures

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is part of academic research. Please cite appropriately if used in publications.

## Acknowledgments

- FER2013 dataset for facial expression recognition
- Hugging Face Transformers for TER models
- OpenCV community for computer vision tools
- Speech Recognition library contributors

---

For questions or issues, please create an issue in the repository or contact me.
