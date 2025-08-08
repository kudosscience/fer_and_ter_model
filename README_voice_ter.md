# Voice-based Textual Emotion Recognition (TER) Inference

This script provides a complete voice-to-emotion analysis pipeline using a trained DistilBERT model for Textual Emotion Recognition. It captures voice input, transcribes it to text, and predicts emotions based on Ekman's basic emotions.

## 🎯 Features

- **Real-time Voice Capture**: Uses microphone to capture voice input
- **Speech Transcription**: Converts speech to text using Google Speech Recognition
- **Emotion Prediction**: Classifies text into 7 basic emotions using trained DistilBERT
- **Interactive Mode**: Continuous voice-to-emotion analysis with real-time feedback
- **Batch Processing**: Process text files with multiple sentences
- **Comprehensive Analytics**: View prediction history and emotion statistics
- **Multiple Input Methods**: Voice, typed text, or file input
- **Device Flexibility**: Auto-detect or force CPU/GPU usage

## 🎭 Supported Emotions

The system recognizes **Ekman's 7 basic emotions**:
- **Angry** 😠
- **Disgust** 🤢  
- **Fear** 😨
- **Happy** 😊
- **Neutral** 😐
- **Sad** 😢
- **Surprise** 😲

## 📋 Requirements

### System Requirements
- Python 3.7+
- Microphone (for voice input)
- Internet connection (for Google Speech Recognition)

### Python Packages
Install all dependencies with:
```bash
pip install -r requirements_voice_ter.txt
```

**Core packages:**
- `torch` - PyTorch for model inference
- `transformers` - Hugging Face transformers for DistilBERT
- `speechrecognition` - Speech-to-text conversion
- `pyaudio` - Audio capture from microphone
- `scikit-learn` - Label encoding and preprocessing
- `numpy` - Numerical operations

### Platform-specific Notes

**Windows:**
```bash
# PyAudio might need conda installation
conda install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

## 🚀 Quick Start

### 1. Interactive Voice Mode (Recommended)
```bash
python voice_ter_inference.py
```
This starts the interactive mode where you can:
- Press ENTER to capture voice and analyze emotion
- Type commands for different operations
- View prediction history and statistics

### 2. Analyze Specific Text
```bash
python voice_ter_inference.py --text "I am feeling amazing today!"
```

### 3. Process Text File
```bash
python voice_ter_inference.py --file input.txt --output results.json
```

### 4. Demo and Examples
```bash
python demo_voice_ter.py
```

## 📖 Detailed Usage

### Command Line Arguments

```bash
python voice_ter_inference.py [OPTIONS]
```

**Options:**
- `--model PATH` - Path to trained model directory (auto-detects if not specified)
- `--device {auto,cpu,cuda}` - Computation device (default: auto)
- `--text TEXT` - Analyze specific text and exit
- `--file PATH` - Process text file and exit
- `--output PATH` - Output file for batch processing results
- `--max-length INT` - Maximum sequence length for tokenization (default: 128)

### Interactive Mode Commands

When running in interactive mode, you can use these commands:

| Command | Description |
|---------|-------------|
| `ENTER` | Start voice capture and emotion analysis |
| `text: <message>` | Analyze typed text directly |
| `history` | Show recent prediction history |
| `stats` | Show emotion statistics for current session |
| `quit` / `exit` / `q` | Exit the application |

### Examples

**Basic voice analysis:**
```bash
python voice_ter_inference.py
# Press ENTER when prompted, then speak
# Example: "I'm really excited about this project!"
# Output: HAPPY (confidence: 0.892)
```

**Batch text analysis:**
```bash
# Create input.txt with multiple lines
echo "I love this weather!" > input.txt
echo "This makes me so angry!" >> input.txt
echo "I'm scared about tomorrow." >> input.txt

# Process the file
python voice_ter_inference.py --file input.txt --output emotions.json
```

**Force CPU usage:**
```bash
python voice_ter_inference.py --device cpu
```

## 🏗️ Model Structure

The system expects a trained DistilBERT model with the following structure:
```
ter_distilbert_model/
├── config.json                 # Model configuration
├── pytorch_model.bin           # Model weights
├── tokenizer.json              # Tokenizer
├── tokenizer_config.json       # Tokenizer configuration
├── vocab.txt                   # Vocabulary
├── label_encoder.pkl           # Label encoder for emotions
├── training_config.pkl         # Training configuration
└── enhanced_*.pkl              # Training datasets (optional)
```

**Auto-detection paths:**
- `./ter_distilbert_model`
- `./ter-model`
- `./models/ter_distilbert_model`

If no local model is found, the system falls back to a pre-trained DistilBERT model.

## 📊 Output Format

### Single Prediction
```json
{
  "text": "I am so happy today!",
  "cleaned_text": "i am so happy today!",
  "predicted_emotion": "happy",
  "confidence": 0.892,
  "timestamp": "2025-08-08T14:30:00.123456",
  "all_probabilities": {
    "happy": 0.892,
    "surprise": 0.045,
    "neutral": 0.032,
    "angry": 0.015,
    "sad": 0.010,
    "fear": 0.004,
    "disgust": 0.002
  }
}
```

### Batch Processing Output
```json
[
  {
    "text": "I love this weather!",
    "predicted_emotion": "happy",
    "confidence": 0.856,
    "timestamp": "2025-08-08T14:30:01.000000"
  },
  {
    "text": "This makes me so angry!",
    "predicted_emotion": "angry", 
    "confidence": 0.923,
    "timestamp": "2025-08-08T14:30:02.000000"
  }
]
```

## 🔧 Troubleshooting

### Common Issues

**1. Microphone not detected:**
```bash
# Check audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); print(f'Audio devices: {p.get_device_count()}'); p.terminate()"
```

**2. PyAudio installation issues:**
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

**3. CUDA/GPU issues:**
```bash
# Force CPU usage
python voice_ter_inference.py --device cpu
```

**4. Speech recognition not working:**
- Check internet connection (Google Speech Recognition requires internet)
- Ensure microphone permissions are granted
- Try speaking more clearly and closer to microphone
- Check microphone volume levels

**5. Model loading errors:**
- Ensure the model directory exists and contains all required files
- Check file permissions
- Try running the demo first: `python demo_voice_ter.py`

### Performance Tips

**1. Improve speech recognition accuracy:**
- Speak clearly and at normal pace
- Use a good quality microphone
- Minimize background noise
- Adjust microphone input levels

**2. Optimize model performance:**
- Use GPU if available (automatic detection)
- Reduce max_length for shorter texts
- Close other GPU-intensive applications

**3. Better emotion prediction:**
- Use complete sentences rather than single words
- Provide context in your speech
- Ensure audio is clear for accurate transcription

## 📁 File Structure

```
fer_and_ter_model/
├── voice_ter_inference.py          # Main inference script
├── demo_voice_ter.py               # Demo and examples
├── requirements_voice_ter.txt       # Dependencies
├── README_voice_ter.md             # This file
├── ter_distilbert_model/           # Trained model (if available)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── label_encoder.pkl
└── textual_emotion_recognition_distilbert.ipynb  # Training notebook
```

## 🎓 Training Your Own Model

To train your own TER model, use the provided Jupyter notebook:
```bash
jupyter notebook textual_emotion_recognition_distilbert.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model training with DistilBERT
- Evaluation and metrics
- Model saving for inference

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs and issues
- Suggesting new features
- Improving documentation
- Adding support for new speech recognition engines
- Optimizing performance

## 📄 License

This project is part of the FER and TER Model project for academic research purposes.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **Google** for Speech Recognition API
- **PyAudio** for audio capture
- **Ekman** for emotion theory foundation

---

**Happy Emotion Analysis!** 🎭✨
