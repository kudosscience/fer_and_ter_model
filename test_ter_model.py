import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

print('Testing DistilBERT wrapper model...')

# Define the custom wrapper class
class DistilBERTWrapper(tf.keras.Model):
    def __init__(self, distilbert_model, num_classes):
        super().__init__()
        self.distilbert = distilbert_model.distilbert
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.features = tf.keras.layers.Dense(64, activation='relu', name='ter_features')
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(7, activation='softmax', name='ter_output')
        
    def call(self, inputs, training=None):
        input_ids, attention_mask = inputs
        
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )
        
        # Use CLS token (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply custom layers
        x = self.dense1(pooled_output)
        x = self.dropout(x, training=training)
        x = self.features(x)
        x = self.dropout2(x, training=training) 
        output = self.classifier(x)
        
        return output

# Test the model creation
print('Loading DistilBERT model...')
base_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=7)
print('Creating wrapper model...')
model = DistilBERTWrapper(base_model, 7)

# Build the model with dummy inputs
print('Testing with dummy inputs...')
dummy_input_ids = tf.zeros((1, 128), dtype=tf.int32)
dummy_attention_mask = tf.ones((1, 128), dtype=tf.int32)
output = model([dummy_input_ids, dummy_attention_mask])

print(f'Model created successfully! Output shape: {output.shape}')
print('KerasTensor error resolved!')

# Test compilation
print('Testing model compilation...')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Model compiled successfully!')
