import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TFDistilBertModel, DistilBertTokenizer, TFDistilBertForSequenceClassification
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json

# Emotion mapping (ensure consistency across modalities)
EMOTION_LABELS = ['joy', 'anger', 'disgust', 'sadness', 'fear', 'surprise']
EMOTION_ID_MAP = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}

# ============== FACIAL EMOTION RECOGNITION (FER) ==============

class FERModel:
    """Facial Emotion Recognition using CNN"""
    
    def __init__(self, input_shape=(48, 48, 1), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._create_model()
        
    def _create_model(self):
        """Creates a lightweight CNN model for emotion recognition."""
        model = tf.keras.Sequential([
            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                         input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Block 3
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fully Connected Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu', name='fer_features'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation='softmax', name='fer_output')
        ])
        
        return model
    
    def get_feature_extractor(self):
        """Returns model that outputs feature embeddings instead of predictions"""
        return tf.keras.Model(inputs=self.model.input, 
                          outputs=self.model.get_layer('fer_features').output)

# ============== TEXT EMOTION RECOGNITION (TER) ==============

class TERModel:
    """Text Emotion Recognition using DistilBERT"""
    
    def __init__(self, num_classes=6, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = self._create_model()
        
    def _create_model(self):
        """Creates DistilBERT-based model for text emotion recognition"""
        # Input layers
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        # DistilBERT base
        distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        distilbert.trainable = True  # Fine-tune the model
        
        # Get BERT outputs
        bert_outputs = distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)
        
        # Additional layers for emotion classification
        x = tf.keras.layers.Dense(256, activation='relu')(cls_output)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu', name='ter_features')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='ter_output')(x)
        
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
        return model
    
    def get_feature_extractor(self):
        """Returns model that outputs feature embeddings instead of predictions"""
        return tf.keras.Model(inputs=self.model.inputs, 
                          outputs=self.model.get_layer('ter_features').output)
    
    def preprocess_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Tokenize and prepare texts for model input"""
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='np'
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

# ============== MULTIMODAL FUSION MODEL ==============

class MultimodalEmotionFusion:
    """Fusion model combining FER and TER"""
    
    def __init__(self, fer_model: FERModel, ter_model: TERModel, 
                 fusion_type: str = 'late', num_classes: int = 6):
        self.fer_model = fer_model
        self.ter_model = ter_model
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.fusion_model = self._create_fusion_model()
        
    def _create_fusion_model(self):
        """Create the fusion model based on fusion type"""
        if self.fusion_type == 'late':
            return self._create_late_fusion_model()
        elif self.fusion_type == 'early':
            return self._create_early_fusion_model()
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def _create_late_fusion_model(self):
        """Late fusion: Combine predictions from both models"""
        # FER inputs
        fer_input = layers.Input(shape=(48, 48, 1), name='image_input')
        
        # TER inputs
        ter_input_ids = layers.Input(shape=(128,), dtype=tf.int32, name='text_input_ids')
        ter_attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name='text_attention_mask')
        
        # Get predictions from both models
        fer_predictions = self.fer_model.model(fer_input)
        ter_predictions = self.ter_model.model([ter_input_ids, ter_attention_mask])
        
        # Fusion layer - weighted average
        fer_weight = layers.Dense(1, activation='sigmoid', name='fer_weight')(fer_predictions)
        ter_weight = layers.Dense(1, activation='sigmoid', name='ter_weight')(ter_predictions)
        
        # Normalize weights
        total_weight = fer_weight + ter_weight
        fer_weight_norm = fer_weight / total_weight
        ter_weight_norm = ter_weight / total_weight
        
        # Weighted fusion
        fusion_output = (fer_predictions * fer_weight_norm + 
                        ter_predictions * ter_weight_norm)
        
        model = tf.keras.Model(
            inputs=[fer_input, ter_input_ids, ter_attention_mask],
            outputs=fusion_output,
            name='late_fusion_model'
        )
        return model
    
    def _create_early_fusion_model(self):
        """Early fusion: Combine features from both models"""
        # FER inputs
        fer_input = tf.keras.layers.Input(shape=(48, 48, 1), name='image_input')
        
        # TER inputs
        ter_input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='text_input_ids')
        ter_attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='text_attention_mask')
        
        # Get feature extractors
        fer_extractor = self.fer_model.get_feature_extractor()
        ter_extractor = self.ter_model.get_feature_extractor()
        
        # Extract features
        fer_features = fer_extractor(fer_input)
        ter_features = ter_extractor([ter_input_ids, ter_attention_mask])
        
        # Concatenate features
        combined_features = tf.keras.layers.Concatenate()([fer_features, ter_features])
        
        # Fusion layers
        x = tf.keras.layers.Dense(128, activation='relu')(combined_features)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        fusion_output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(
            inputs=[fer_input, ter_input_ids, ter_attention_mask],
            outputs=fusion_output,
            name='early_fusion_model'
        )
        return model

# ============== DATA LOADING AND PREPROCESSING ==============

class MultimodalDataset:
    """Handles multimodal dataset with images and text"""
    
    def __init__(self, data_path: str, tokenizer: DistilBertTokenizer, max_length: int = 128):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load multimodal dataset from JSON file"""
        # Expected format: [{"image_path": "...", "text": "...", "emotion": "..."}]
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for FER model"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=-1)
    
    def prepare_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Prepare all data for training"""
        # Process images
        images = np.array([self.preprocess_image(path) for path in self.data['image_path']])
        
        # Process texts
        text_inputs = self.tokenizer(
            self.data['text'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='np'
        )
        
        # Process labels
        labels = np.array([EMOTION_ID_MAP[emotion] for emotion in self.data['emotion']])
        labels_categorical = tf.keras.utils.to_categorical(labels, num_classes=6)
        
        return {
            'images': images,
            'text_input_ids': text_inputs['input_ids'],
            'text_attention_mask': text_inputs['attention_mask']
        }, labels_categorical

# ============== TRAINING AND EVALUATION ==============

class ModelTrainer:
    """Handles training and evaluation of all models"""
    
    def __init__(self, fer_model: FERModel, ter_model: TERModel, 
                 fusion_model: MultimodalEmotionFusion):
        self.fer_model = fer_model
        self.ter_model = ter_model
        self.fusion_model = fusion_model
        self.history = {}
        
    def train_fer_model(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """Train FER model"""
        print("\n" + "="*50)
        print("Training FER Model")
        print("="*50)
        
        self.fer_model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.fer_model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.history['fer'] = history
        return history
    
    def train_ter_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=16):
        """Train TER model"""
        print("\n" + "="*50)
        print("Training TER Model")
        print("="*50)
        
        self.ter_model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.ter_model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.history['ter'] = history
        return history
    
    def train_fusion_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
        """Train fusion model"""
        print("\n" + "="*50)
        print("Training Fusion Model")
        print("="*50)
        
        # Freeze base models initially
        self.fer_model.model.trainable = False
        self.ter_model.model.trainable = False
        
        self.fusion_model.fusion_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train fusion layers
        history = self.fusion_model.fusion_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs//2,
            batch_size=batch_size,
            verbose=1
        )
        
        # Unfreeze and fine-tune
        self.fer_model.model.trainable = True
        self.ter_model.model.trainable = True
        
        self.fusion_model.fusion_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history_finetune = self.fusion_model.fusion_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs//2,
            batch_size=batch_size,
            verbose=1,
            initial_epoch=epochs//2
        )
        
        # Combine histories
        for key in history.history:
            history.history[key].extend(history_finetune.history[key])
        
        self.history['fusion'] = history
        return history

# ============== EVALUATION AND COMPARISON ==============

class ModelEvaluator:
    """Comprehensive evaluation and comparison of models"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_fer(self, model: FERModel, X_test, y_test):
        """Evaluate FER model"""
        predictions = model.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True)
        
        self.results['fer'] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': predictions
        }
        
        return accuracy
    
    def evaluate_ter(self, model: TERModel, X_test, y_test):
        """Evaluate TER model"""
        predictions = model.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True)
        
        self.results['ter'] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': predictions
        }
        
        return accuracy
    
    def evaluate_fusion(self, model: MultimodalEmotionFusion, X_test, y_test):
        """Evaluate fusion model"""
        predictions = model.fusion_model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True)
        
        self.results['fusion'] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': predictions
        }
        
        return accuracy
    
    def compare_models(self):
        """Generate comprehensive comparison of all models"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        # Accuracy comparison
        print("\nAccuracy Scores:")
        print("-" * 40)
        for model_name, results in self.results.items():
            print(f"{model_name.upper():10s}: {results['accuracy']:.4f}")
        
        # Improvement calculation
        if 'fer' in self.results and 'ter' in self.results and 'fusion' in self.results:
            fer_acc = self.results['fer']['accuracy']
            ter_acc = self.results['ter']['accuracy']
            fusion_acc = self.results['fusion']['accuracy']
            
            best_single = max(fer_acc, ter_acc)
            improvement = ((fusion_acc - best_single) / best_single) * 100
            
            print(f"\nFusion Model Improvement: {improvement:.2f}%")
        
        # Per-class performance
        print("\nPer-Class F1 Scores:")
        print("-" * 60)
        print(f"{'Emotion':12s} | {'FER':8s} | {'TER':8s} | {'Fusion':8s}")
        print("-" * 60)
        
        for emotion in EMOTION_LABELS:
            fer_f1 = self.results.get('fer', {}).get('classification_report', {}).get(emotion, {}).get('f1-score', 0)
            ter_f1 = self.results.get('ter', {}).get('classification_report', {}).get(emotion, {}).get('f1-score', 0)
            fusion_f1 = self.results.get('fusion', {}).get('classification_report', {}).get(emotion, {}).get('f1-score', 0)
            print(f"{emotion:12s} | {fer_f1:8.4f} | {ter_f1:8.4f} | {fusion_f1:8.4f}")
        
        return self.results
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (model_name, ax) in enumerate(zip(['fer', 'ter', 'fusion'], axes)):
            if model_name in self.results:
                cm = self.results[model_name]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
                ax.set_title(f'{model_name.upper()} Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.show()
    
    def plot_performance_comparison(self):
        """Plot performance comparison bar chart"""
        if not self.results:
            return
        
        # Prepare data
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'coral'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig('performance_comparison.png')
        plt.show()

# ============== REAL-TIME MULTIMODAL EMOTION DETECTION ==============

class MultimodalEmotionDetector:
    """Real-time emotion detection using both face and text"""
    
    def __init__(self, fusion_model: MultimodalEmotionFusion):
        self.fusion_model = fusion_model
        self.fer_model = fusion_model.fer_model
        self.ter_model = fusion_model.ter_model
        self.tokenizer = fusion_model.ter_model.tokenizer
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Text buffer for continuous emotion detection
        self.text_buffer = ""
        self.last_text_update = 0
        
    def preprocess_face(self, face_img):
        """Preprocess face for FER model"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype('float32') / 255.0
        return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
    
    def predict_multimodal(self, face_img, text):
        """Predict emotion using both face and text"""
        # Preprocess face
        face_input = self.preprocess_face(face_img)
        
        # Preprocess text
        text_inputs = self.ter_model.preprocess_texts([text])
        
        # Get individual predictions
        fer_pred = self.fer_model.model.predict(face_input, verbose=0)[0]
        ter_pred = self.ter_model.model.predict(
            [text_inputs['input_ids'], text_inputs['attention_mask']], verbose=0)[0]
        
        # Get fusion prediction
        fusion_pred = self.fusion_model.fusion_model.predict(
            [face_input, text_inputs['input_ids'], text_inputs['attention_mask']], verbose=0)[0]
        
        # Get emotion labels and confidences
        fer_emotion = EMOTION_LABELS[np.argmax(fer_pred)]
        ter_emotion = EMOTION_LABELS[np.argmax(ter_pred)]
        fusion_emotion = EMOTION_LABELS[np.argmax(fusion_pred)]
        
        return {
            'fer': {'emotion': fer_emotion, 'confidence': np.max(fer_pred), 'probs': fer_pred},
            'ter': {'emotion': ter_emotion, 'confidence': np.max(ter_pred), 'probs': ter_pred},
            'fusion': {'emotion': fusion_emotion, 'confidence': np.max(fusion_pred), 'probs': fusion_pred}
        }

# ============== EXAMPLE USAGE ==============

def create_sample_multimodal_dataset():
    """Create a sample dataset structure for demonstration"""
    sample_data = [
        {
            "image_path": "path/to/happy_face1.jpg",
            "text": "I'm feeling great today! Everything is going wonderfully.",
            "emotion": "joy"
        },
        {
            "image_path": "path/to/angry_face1.jpg",
            "text": "This is absolutely unacceptable! I'm furious about this.",
            "emotion": "anger"
        },
        {
            "image_path": "path/to/sad_face1.jpg",
            "text": "I feel so down and hopeless. Nothing seems to work out.",
            "emotion": "sadness"
        },
        # Add more samples...
    ]
    
    # Save to JSON
    with open('multimodal_emotion_dataset.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample dataset structure created: multimodal_emotion_dataset.json")

def main():
    """Main execution pipeline"""
    print("Multimodal Emotion Recognition System")
    print("="*60)
    
    # 1. Initialize models
    fer_model = FERModel()
    ter_model = TERModel()
    fusion_model = MultimodalEmotionFusion(fer_model, ter_model, fusion_type='late')
    
    # 2. Print model summaries
    print("\nFER Model Summary:")
    fer_model.model.summary()
    
    print("\nTER Model Summary:")
    ter_model.model.summary()
    
    print("\nFusion Model Summary:")
    fusion_model.fusion_model.summary()
    
    # 3. Calculate total parameters
    fer_params = fer_model.model.count_params()
    ter_params = ter_model.model.count_params()
    fusion_params = fusion_model.fusion_model.count_params()
    
    print(f"\nModel Sizes:")
    print(f"FER Model: {fer_params:,} parameters (~{fer_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"TER Model: {ter_params:,} parameters (~{ter_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"Fusion Model: {fusion_params:,} parameters (~{fusion_params * 4 / 1024 / 1024:.2f} MB)")
    
    # Example training pipeline (commented out - requires actual data)
    """
    # Load multimodal dataset
    dataset = MultimodalDataset('multimodal_emotion_dataset.json', ter_model.tokenizer)
    data, labels = dataset.prepare_data()
    
    # Split data
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
    
    # Prepare training data
    X_train_fer = data['images'][train_idx]
    X_train_ter = [data['text_input_ids'][train_idx], data['text_attention_mask'][train_idx]]
    X_train_fusion = [X_train_fer, X_train_ter[0], X_train_ter[1]]
    y_train = labels[train_idx]
    
    # Initialize trainer
    trainer = ModelTrainer(fer_model, ter_model, fusion_model)
    
    # Train individual models
    trainer.train_fer_model(X_train_fer, y_train, X_val_fer, y_val)
    trainer.train_ter_model(X_train_ter, y_train, X_val_ter, y_val)
    trainer.train_fusion_model(X_train_fusion, y_train, X_val_fusion, y_val)
    
    # Evaluate and compare
    evaluator = ModelEvaluator()
    evaluator.evaluate_fer(fer_model, X_test_fer, y_test)
    evaluator.evaluate_ter(ter_model, X_test_ter, y_test)
    evaluator.evaluate_fusion(fusion_model, X_test_fusion, y_test)
    
    # Generate comparison report
    evaluator.compare_models()
    evaluator.plot_confusion_matrices()
    evaluator.plot_performance_comparison()
    """
    
    # Create sample dataset structure
    create_sample_multimodal_dataset()

if __name__ == "__main__":
    main()