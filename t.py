import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# Load the YAMNet model from TensorFlow Hub
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# Function to load and preprocess audio data for YAMNet
def load_audio_for_yamnet(file_path, sample_rate=16000):
    audio_data, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio_data

# Data Augmentation Functions
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def time_stretch(audio, rate=1.0):
    return librosa.effects.time_stretch(audio, rate)

def pitch_shift(audio, sample_rate, pitch_factor=0):
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_factor)

# Augment the audio by applying random pitch shifting, time stretching, and adding noise
def augment_audio(audio, sample_rate):
    if np.random.random() > 0.5:  # 50% chance to apply pitch shift
        audio = pitch_shift(audio, sample_rate, np.random.randint(-3, 3))
    
    if np.random.random() > 0.5:  # 50% chance to apply time stretch
        audio = time_stretch(audio, np.random.uniform(0.8, 1.2))  # Correct usage
    
    if np.random.random() > 0.5:  # 50% chance to add noise
        audio = add_noise(audio)
    
    return audio

# Function to extract YAMNet embeddings
def extract_yamnet_embeddings(audio_data):
    # YAMNet expects a waveform with shape (None,)
    # Reshape to a 1D array if required
    if len(audio_data.shape) > 1:
        audio_data = np.squeeze(audio_data)
    
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    return embeddings.numpy()  # Convert to numpy array for training

# Load and preprocess your dataset
def load_dataset_with_embeddings(dataset_path, sample_rate=16000, augment=False):
    X = []
    y = []
    audio_formats = ('.webm', '.wav', '.mp3')  # Supported audio formats

    print(f"Checking contents of dataset path: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Directory {dataset_path} does not exist!")
        return [], []

    for label in os.listdir(dataset_path):  # Each folder is treated as a class
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            print(f"Skipping non-directory item: {label_path}")
            continue  # Skip non-directory entries

        print(f"Processing class: {label}")
        print(f"Contents of {label_path}: {os.listdir(label_path)}")

        for audio_file in os.listdir(label_path):
            file_path = os.path.join(label_path, audio_file)
            print(f"Attempting to load file: {file_path}")
            
            if not audio_file.endswith(audio_formats):  # Skip non-audio files
                print(f"Skipping non-audio file: {audio_file}")
                continue
            
            try:
                # Load audio and extract embeddings
                audio_data = load_audio_for_yamnet(file_path, sample_rate)
                
                # Apply data augmentation if augment=True
                if augment:
                    audio_data = augment_audio(audio_data, sample_rate)
                
                embeddings = extract_yamnet_embeddings(audio_data)
                
                X.append(embeddings.mean(axis=0))  # Take mean across time dimension
                y.append(label)  # The label is the folder name
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Improved classifier model for YAMNet embeddings
def create_yamnet_classifier(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Set the path to the already extracted dataset
dataset_path = 'F:\\Audio_project\\Data'  # Use the correct path to your data

# Check if the dataset folder exists
if not os.path.exists(dataset_path):
    print(f"Folder not found in {dataset_path}. Make sure the dataset path is correct.")
    raise FileNotFoundError(f"Dataset folder {dataset_path} not found.")

# Load dataset and extract embeddings using YAMNet, apply augmentation if needed
X, y = load_dataset_with_embeddings(dataset_path, augment=True)

# Check if we have any data
if len(X) == 0 or len(y) == 0:
    print(f"ERROR: No data found in {dataset_path}.")
    raise ValueError("No data found in the dataset.")

# Encode labels to one-hot (ensure correct shape)
label_binarizer = OneHotEncoder(sparse_output=False)
y = label_binarizer.fit_transform(y.reshape(-1, 1))  # Reshape for encoding to 2D

num_classes = y.shape[1]  # Adjust based on actual number of classes

# Debug print shapes
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", num_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the input shape for the classifier model
input_shape = (X_train.shape[1],)

# Create and train the classifier
model = create_yamnet_classifier(input_shape, num_classes)

# Train the model with early stopping and validation
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix and Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_binarizer.categories_[0]))

# Save the classifier model
model.save('yamnet_audio_classifier_augmented.h5')

print("YAMNet-based audio classifier training complete with data augmentation.")
