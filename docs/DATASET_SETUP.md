# Quick Start Guide for Dataset Setup

## 📁 Directory Structure

```
📦 fer_and_ter_model/
├── 📁 data/                     # Main data directory
│   ├── 📄 README.md             # Detailed documentation
│   ├── 📁 raw/                  # Raw, unprocessed data
│   │   ├── 📁 fer_images/       # Facial emotion images
│   │   │   ├── 📁 train/        # Training images
│   │   │   │   ├── 📁 joy/      # Joy emotion images
│   │   │   │   ├── 📁 anger/    # Anger emotion images  
│   │   │   │   ├── 📁 disgust/  # Disgust emotion images
│   │   │   │   ├── 📁 sadness/  # Sadness emotion images
│   │   │   │   ├── 📁 fear/     # Fear emotion images
│   │   │   │   └── 📁 surprise/ # Surprise emotion images
│   │   │   ├── 📁 test/         # Test images (same structure)
│   │   │   └── 📁 validation/   # Validation images (same structure)
│   │   ├── 📁 text_data/        # Text emotion data
│   │   └── 📁 multimodal_data/  # Combined image+text data
│   └── 📁 processed/            # Preprocessed features
│       ├── 📁 fer_features/     # Extracted FER features
│       └── 📁 ter_features/     # Extracted TER features
├── 📁 models/                   # Saved trained models
├── 📁 results/                  # Training results and metrics
│   ├── 📁 plots/               # Generated plots and visualizations
│   └── 📁 metrics/             # Performance metrics
└── 📄 data_utils.py            # Dataset management utilities
```

## 🚀 Getting Started

### Step 1: Add Your Images
Place your facial emotion images in the appropriate directories:
- `data/raw/fer_images/train/joy/` - for joyful face images
- `data/raw/fer_images/train/anger/` - for angry face images  
- etc.

### Step 2: Create Text Data
Create JSON files following this format:
```json
[
  {
    "text": "I'm feeling great today!",
    "emotion": "joy",
    "id": "text_001"
  }
]
```

### Step 3: Create Multimodal Data
Link images with text using this format:
```json
[
  {
    "image_path": "data/raw/fer_images/train/joy/happy_face.jpg",
    "text": "I'm feeling great today!",
    "emotion": "joy", 
    "id": "multimodal_001"
  }
]
```

### Step 4: Use the Dataset Manager
Run the data utilities to check your dataset:
```bash
python data_utils.py
```

## 📊 Emotion Classes
The system supports 6 emotions:
- **joy** 😊 - Happy, positive emotions
- **anger** 😠 - Angry, frustrated emotions  
- **disgust** 🤢 - Disgusted, repulsed emotions
- **sadness** 😢 - Sad, depressed emotions
- **fear** 😨 - Fearful, anxious emotions
- **surprise** 😲 - Surprised, shocked emotions

## 💡 Tips for Success

1. **Balance Your Data**: Aim for equal numbers of samples per emotion
2. **Quality Over Quantity**: Better to have fewer high-quality samples
3. **Test Split**: Keep 10-20% of data for final testing
4. **Validation Split**: Use 10-15% for hyperparameter tuning

## 📚 Popular Datasets You Can Use

### Facial Emotion Recognition:
- **FER2013** - Classic facial emotion dataset
- **AffectNet** - Large-scale facial expressions
- **RAF-DB** - Real-world affective faces

### Text Emotion Recognition:
- **GoEmotions** - Google's emotion dataset  
- **SemEval-2018** - Emotion classification tasks
- **ISEAR** - International emotion survey

## 🔧 Next Steps

1. **Collect/Download** your chosen dataset
2. **Organize** files into the created directory structure
3. **Run** `python data_utils.py` to validate your setup
4. **Train** your models using `python emotion-cnn-model.py`

## 📞 Need Help?

- Check `data/README.md` for detailed documentation
- Use `data_utils.py` to manage and validate your dataset
- The system will automatically preprocess your data during training
