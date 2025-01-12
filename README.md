# YAMNet-Based-Audio-Classifier

# Noise Detection and Classification in Real-time for Industrial Environments

This Streamlit application leverages real-time audio processing to detect and classify noise levels in industrial environments, ensuring machinery safety. It utilizes TensorFlow, YAMNet—a deep learning model trained to recognize 521 audio events—and additional custom model layers to identify audio patterns indicative of normal or anomalous conditions.

## Features

- Real-time audio recording via browser interface.
- Audio classification using TensorFlow with pre-trained YAMNet embeddings.
- Visualization of audio waveforms.
- Display of prediction results including class labels and confidence scores.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have Python 3.8+ and pip installed on your system.

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2. **Setup a virtual environment** (recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**

    ```bash
    streamlit run app.py
    ```

Access the application by navigating to `http://localhost:8501` in your web browser.

## How to Use

- **Record Audio**: Set the duration in the sidebar and click "Record Audio" to capture audio through your microphone.
- **Review Output**: Post recording, the application displays the audio waveform and classifies the sound, providing predictions and confidence levels.


## Acknowledgments

- TensorFlow Hub for providing the YAMNet model.
- Streamlit for the intuitive app development interface.
