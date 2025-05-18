import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import os
import pathlib
# --- Force CPU Usage (to bypass CUDA compatibility issues) ---
# Add these lines at the very beginning
try:
    # Hide GPU devices from TensorFlow
    tf.config.set_visible_devices([], 'GPU')
    print("GPU hidden. TensorFlow will use CPU.")
except Exception as e:
    print(f"Could not hide GPU, proceeding potentially with GPU: {e}")
# --- End of CPU Force ---
# --- Configuration ---
TRAIN_DIR = '/home/nax/Masaüstü/image-classification/imagenet_50/train' # Path to the training data directory
TEST_DIR = '/home/nax/Masaüstü/image-classification/imagenet_50/test/imgs'  # Path to the test images directory
SUBMISSION_FILE = 'submission.csv' # Name of the output submission file
IMG_HEIGHT = 128 # Target image height
IMG_WIDTH = 128  # Target image width
BATCH_SIZE = 32   # Batch size for training and predictioncle
EPOCHS = 30       # Number of training epochs (adjust as needed)
AUTOTUNE = tf.data.AUTOTUNE # Optimize data loading performance

# --- 1. Data Loading and Preprocessing (Training) ---
print("Loading training data...")
# Convert train_dir string to Path object for compatibility
train_dir_path = pathlib.Path(TRAIN_DIR)

# Check if the training directory exists
if not train_dir_path.exists():
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

# Use image_dataset_from_directory for efficient loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir_path,
    labels='inferred',       # Infer labels from directory structure
    label_mode='int',        # Use integer labels for SparseCategoricalCrossentropy
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest', # Use 'nearest' for potential speed up, 'bilinear' is default
    batch_size=BATCH_SIZE,
    shuffle=True,            # Shuffle training data
    seed=123                 # Set seed for reproducibility
    # No validation split needed as per the problem description
)

# Get class names (folder names) - crucial for submission file
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names[:5]}...") # Print first 5 classes

# --- 2. Data Augmentation ---
# Define augmentation layers
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        # Add more augmentations if needed (e.g., RandomBrightness)
    ],
    name="data_augmentation",
)

# --- 3. Data Pipeline Configuration (Normalization & Augmentation) ---
# Function to normalize pixel values to [0, 1]
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

# Apply normalization and augmentation to the training dataset
print("Configuring training data pipeline...")
train_ds = train_ds.map(normalize_img, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) # Cache and prefetch for performance

# --- 4. Model Building (Custom CNN from Scratch) ---
print("Building the model...")
def build_custom_cnn(input_shape, num_classes):
    """Builds a custom CNN model layer by layer."""
    model = keras.Sequential(name="custom_cnn")
    model.add(layers.Input(shape=input_shape)) # Define input layer explicitly

    # --- Convolutional Base ---
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 4 
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # --- Classifier Head ---
    model.add(layers.Flatten())
    model.add(layers.Dense(512)) # Densely connected layer
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5)) # Dropout for regularization
    model.add(layers.Dense(num_classes, activation='softmax')) # Output layer

    return model

# Instantiate the model
model = build_custom_cnn((IMG_HEIGHT, IMG_WIDTH, 3), num_classes)
model.summary() # Print model architecture

# --- 5. Model Compilation ---
print("Compiling the model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), # Start with a smaller learning rate
    loss='sparse_categorical_crossentropy', # Use sparse version as labels are integers
    metrics=['accuracy']
)

# --- 6. Training ---
print(f"Starting training for {EPOCHS} epochs...")
# Add callbacks (optional but recommended)
callbacks = [
    # keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1), # Stop if loss doesn't improve
    keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=False, monitor='loss', verbose=1) # Save model periodically
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1 # Show progress bar
)

print("Training finished.")

# --- 7. Prediction on Test Set ---
print("Loading and predicting on test data...")
test_dir_path = pathlib.Path(TEST_DIR)

# Check if the test directory exists
if not test_dir_path.exists():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

# Get list of test image filenames
test_image_paths = sorted([str(p) for p in test_dir_path.glob('*.JPEG')]) # Ensure consistent order

if not test_image_paths:
    raise FileNotFoundError(f"No JPEG images found in {TEST_DIR}")

print(f"Found {len(test_image_paths)} test images.")

# Function to load and preprocess a single test image
def load_and_preprocess_test_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    img = tf.cast(img, tf.float32) / 255.0 # Normalize to [0, 1]
    return img

# Create a dataset from test image paths
test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_image_ds = test_path_ds.map(load_and_preprocess_test_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_image_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE) # Batch test data

# Make predictions
predictions = model.predict(test_image_ds, verbose=1)

# Get the index of the highest probability class for each image
predicted_class_indices = np.argmax(predictions, axis=1)

# Map indices back to class names (the folder names like 'n01440764')
predicted_class_names = [class_names[i] for i in predicted_class_indices]

# --- 8. Submission File Generation ---
print(f"Generating submission file: {SUBMISSION_FILE}...")
# Extract filenames from paths
test_filenames = [os.path.basename(p) for p in test_image_paths]

# Create pandas DataFrame
submission_df = pd.DataFrame({
    'FileName': test_filenames,
    'Class': predicted_class_names
})

# Ensure the order matches the test image list (should be correct due to sorting)
# Double check if the test set requires a specific order not based on filename sorting.
# If so, adjust the test_image_paths loading accordingly.

# Save to CSV
submission_df.to_csv(SUBMISSION_FILE, index=False)

print("Submission file created successfully!")
print(submission_df.head())
