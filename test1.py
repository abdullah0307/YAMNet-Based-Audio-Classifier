
# 2024 Industrial project

import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import librosa

# Define sample rate
sample_rate = 16000  # You can adjust this value if needed

# Load the pre-trained models
@st.cache_resource
def load_models():
    model = load_model('yamnet_audio_classifier_augmented.h5')
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    return model, yamnet_model

model, yamnet_model = load_models()

# Record audio from the browser
def record_audio(duration):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio_data)

# Function to extract YAMNet embeddings
def extract_yamnet_embeddings(audio_data, yamnet_model):
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    return embeddings.numpy().mean(axis=0)

# Streamlit layout
st.title("Noise Detection and Classification in Real-time for industrial environments(Machines Safety)")

with st.sidebar:
    st.header("Controls")
    record_duration = st.slider("Record duration (seconds):", min_value=1, max_value=10, value=5)
    if st.button("Record Audio"):
        audio_data = record_audio(record_duration)
        st.session_state['audio_data'] = audio_data

# Process audio data
if 'audio_data' in st.session_state:
    audio_data = st.session_state['audio_data']
    st.write("Audio Recorded. Processing...")
    embeddings = extract_yamnet_embeddings(audio_data, yamnet_model)
    predictions = model.predict(np.reshape(embeddings, (1, 1024)))
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Waveform")
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
        ax.set_title("Recorded Audio Waveform")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
    with col2:
        st.header("Prediction Results")
        st.write(f"Predicted class: {'Anomaly' if predicted_class[0] == 0 else 'Normal'}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
