import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
base_path = "/home/nax/Desktop/blgodev/1/dataset"
train_folders = ['0_AZIZSANCAR', '1_BIYKEMBOZKURT', '2_CAHITARF', 
                '3_CANANDAGDEVIREN', '4_KORAYKAVUKCUOGLU']
test_folder = 'test'

# Parameters for audio processing
SAMPLE_RATE = 22050  # Standard sample rate
DURATION = 5  # All audio files are about 5 seconds
N_MELS = 128  # Number of mel bands
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for FFT
MAX_SAMPLES = SAMPLE_RATE * DURATION

# Function to extract mel spectrogram features
def extract_features(file_path):
    try:
        # Load audio file with fixed length
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad if audio is shorter than expected
        if len(y) < MAX_SAMPLES:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))
        
        # Trim if audio is longer than expected
        y = y[:MAX_SAMPLES]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            n_mels=N_MELS
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        return mel_spec_db
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        # Return zeros if there's an error
        return np.zeros((N_MELS, 1 + int(MAX_SAMPLES // HOP_LENGTH)))

# Prepare training data
X_train = []
y_train = []

print("Processing training data...")
for class_idx, folder in enumerate(train_folders):
    folder_path = os.path.join(base_path, folder)
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            mel_spec = extract_features(file_path)
            X_train.append(mel_spec)
            y_train.append(class_idx)  # Use folder index as class label

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape for CNN input: (samples, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Prepare test data
test_files = []
X_test = []

print("Processing test data...")
test_path = os.path.join(base_path, test_folder)
for filename in tqdm(os.listdir(test_path)):
    if filename.endswith('.wav'):
        file_path = os.path.join(test_path, filename)
        mel_spec = extract_features(file_path)
        X_test.append(mel_spec)
        test_files.append(filename)

# Convert test data to numpy array
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print(f"Test data shape: {X_test.shape}")

# Build the CNN model
def build_model(input_shape, num_classes):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Second convolutional block with residual connection
    input_layer = x
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Residual connection
    residual = layers.Conv2D(128, kernel_size=(1, 1), padding='same')(input_layer)
    x = layers.add([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Third convolutional block with residual connection
    input_layer = x
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Residual connection
    residual = layers.Conv2D(256, kernel_size=(1, 1), padding='same')(input_layer)
    x = layers.add([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Attention mechanism
    attention = layers.Conv2D(1, kernel_size=(1, 1), padding='same')(x)
    attention = layers.Activation('sigmoid')(attention)
    x = layers.multiply([x, attention])
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
input_shape = X_train.shape[1:]
num_classes = len(train_folders)
model = build_model(input_shape, num_classes)

# Use a learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Define callbacks
# Define improved callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor accuracy instead of loss
    patience=15,             # Increased patience
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',  # Monitor accuracy instead of loss
    factor=0.2,              # More aggressive reduction
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Add model checkpoint to save best models
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_sound_classification_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,  # Increase epochs since we have early stopping
    batch_size=32,
    validation_split=0.05,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Make predictions on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Create submission dataframe
submission = pd.DataFrame({
    'FileName': test_files,
    'Class': y_pred
})

# Save submission to CSV
submission_path = os.path.join(base_path, 'submission.csv')
submission.to_csv(submission_path, index=False)

print(f"Submission file saved to: {submission_path}")
print(f"Number of predictions: {len(submission)}")
print(submission.head())