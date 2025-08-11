# Furhat Multimodal Emotion Recognition System

This script integrates the multimodal emotion recognition system with the Furhat robot, using the robot's camera and microphone for emotion detection and providing interactive emotional responses.

## Features

- **Furhat Remote API Integration**: Controls the Furhat robot using the official Python Remote API
- **Multimodal Emotion Detection**: Combines Facial Expression Recognition (FER) and Textual Emotion Recognition (TER)
- **Robot Sensor Input**: Uses Furhat's microphone for speech recognition and text emotion analysis
- **Interactive Robot Responses**: Robot responds with appropriate gestures, speech, and LED colors based on detected emotions
- **Real-time Processing**: Continuous emotion monitoring with configurable fusion strategies
- **Visual Interface**: Live display of emotion predictions and system status

## Prerequisites

### Hardware Requirements
1. **Furhat Robot**: A Furhat robot with Remote API enabled
2. **Network Connection**: Robot and computer must be on the same network

### Software Requirements
1. **Python 3.7+**
2. **Required packages** (install with requirements file)
3. **Trained Models**: FER and TER models (same as original multimodal system)

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_furhat.txt
   ```

2. **Install Furhat Remote API** (if not included in requirements):
   ```bash
   pip install furhat-remote-api
   ```

3. **Ensure Model Files**:
   - `fer2013_final_model.pth` (FER model)
   - `ter_distilbert_model/` directory (TER model)

## Furhat Robot Setup

1. **Enable Remote API**:
   - Start the Furhat robot
   - Open the Furhat web interface
   - Enable the Remote API service (usually on port 54321)

2. **Network Configuration**:
   - Note the robot's IP address
   - Ensure firewall allows connections to port 54321

## Usage

### Basic Usage
```bash
python furhat_multimodal_emotion_inference.py
```

### With Custom Robot IP
```bash
python furhat_multimodal_emotion_inference.py --furhat_ip 192.168.1.100
```

### With Custom Models
```bash
python furhat_multimodal_emotion_inference.py --fer_model ./my_fer_model.pth --ter_model ./my_ter_model
```

### Disable Robot Responses (Detection Only)
```bash
python furhat_multimodal_emotion_inference.py --no_robot_responses
```

### Full Command Line Options
```bash
python furhat_multimodal_emotion_inference.py \
    --fer_model ./fer2013_final_model.pth \
    --ter_model ./ter_distilbert_model \
    --device cuda \
    --furhat_ip 192.168.1.100 \
    --fusion confidence_based \
    --no_robot_responses
```

## Interactive Controls

### Keyboard Controls
- **Q**: Quit the application
- **V**: Toggle Furhat voice capture (start/stop listening)
- **S**: Save current frame as image
- **T**: Toggle TER panel display
- **F**: Toggle fullscreen mode
- **H**: Show prediction history
- **P**: Show emotion statistics
- **R**: Toggle robot emotional responses
- **G**: Test robot gesture manually

### Robot Interactions
- **Voice Input**: Speak to the robot when voice capture is active (press 'V')
- **Emotional Responses**: Robot responds with gestures, speech, and LED colors
- **Real-time Feedback**: Visual display shows current emotion states

## System Architecture

### Input Sources
1. **Furhat Microphone**: Speech recognition via `furhat.listen()`
2. **Furhat Camera**: Visual input (implementation dependent on setup)

### Processing Pipeline
1. **Speech-to-Text**: Furhat converts speech to text
2. **Text Emotion Recognition**: TER model analyzes emotional content
3. **Facial Expression Recognition**: FER model analyzes facial expressions (when camera available)
4. **Emotion Fusion**: Combines FER and TER predictions using configurable strategies
5. **Robot Response**: Generates appropriate gestures, speech, and visual feedback

### Emotion Responses

#### Supported Emotions
- **Happy**: BigSmile gesture, positive speech, yellow LED
- **Sad**: Frown gesture, empathetic speech, blue LED
- **Angry**: Anger gesture, calming speech, red LED
- **Surprise**: Surprise gesture, excited speech, orange LED
- **Fear**: Worry gesture, supportive speech, purple LED
- **Disgust**: Disgust gesture, understanding speech, green LED
- **Neutral**: No gesture, neutral speech, gray LED

#### Response Cooldown
- 3-second cooldown between robot responses to prevent overwhelming behavior
- Configurable response threshold (default: 30% confidence minimum)

## Configuration

### Fusion Strategies
1. **Weighted Average**: Combines FER (60%) and TER (40%) with confidence boosting for agreement
2. **Confidence Based**: Selects the most confident prediction with agreement boosting

### Robot Behavior
- **Gesture Mapping**: Each emotion maps to specific Furhat gestures
- **Speech Responses**: Multiple response options per emotion for variety
- **LED Colors**: Emotion-specific color coding
- **Response Timing**: Configurable cooldown and confidence thresholds

## Troubleshooting

### Connection Issues
1. **Robot Not Found**:
   - Verify Furhat robot is powered on
   - Check Remote API is enabled in robot settings
   - Confirm network connectivity and IP address

2. **API Errors**:
   - Ensure furhat-remote-api is installed correctly
   - Check robot's Remote API service status
   - Verify firewall settings allow port 54321

### Performance Issues
1. **Slow Response**:
   - Check network latency to robot
   - Consider reducing confidence thresholds
   - Disable robot responses for detection-only mode

2. **Model Loading**:
   - Verify model files are in correct locations
   - Check CUDA availability for GPU acceleration
   - Ensure sufficient memory for model loading

### Audio Issues
1. **Speech Recognition**:
   - Test robot's microphone functionality
   - Check ambient noise levels
   - Verify speech recognition language settings

## File Structure
```
├── furhat_multimodal_emotion_inference.py  # Main Furhat integration script
├── requirements_furhat.txt                 # Furhat-specific requirements
├── README_furhat.md                        # This documentation
├── fer2013_final_model.pth                 # FER model file
└── ter_distilbert_model/                   # TER model directory
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── label_encoder.pkl
    └── ...
```

## API Reference

### Furhat Remote API Methods Used
- `furhat.listen()`: Capture speech input
- `furhat.say(text)`: Generate speech output
- `furhat.gesture(name)`: Perform emotional gestures
- `furhat.set_led(red, green, blue)`: Control LED colors
- `furhat.get_users()`: Detect users (for camera input)
- `furhat.get_voices()`: List available voices

### Key Classes
- `FurhatMultimodalEmotionInference`: Main system class
- `EmotionCNN`: Facial expression recognition model
- Emotion fusion and response generation methods

## Limitations

1. **Camera Access**: Direct camera access through Remote API may be limited
2. **Network Dependency**: Requires stable network connection to robot
3. **Response Latency**: Network communication introduces slight delays
4. **Gesture Library**: Limited to predefined Furhat gestures

## Future Enhancements

1. **Enhanced Camera Integration**: Direct camera stream processing
2. **Custom Gesture Creation**: Dynamic gesture generation based on emotions
3. **Multi-User Support**: Handle multiple users simultaneously
4. **Emotion Tracking**: Long-term emotion pattern analysis
5. **Voice Synthesis**: Emotion-aware speech generation

## Support

For issues related to:
- **Furhat Robot**: Contact Furhat Robotics support
- **Remote API**: Check Furhat Developer Documentation
- **Emotion Models**: Refer to original multimodal system documentation
- **Integration Issues**: Check troubleshooting section above

## License

This integration follows the same license as the original multimodal emotion recognition system.
