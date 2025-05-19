# Facial_Emotion_Recognition

This project uses a Convolutional Neural Network (CNN) to detect human emotions from facial images. The model is built using Keras and trained on an image dataset containing labeled emotions.

## 📁 Dataset Structure

Ensure your dataset follows this directory structure:
images/
├── train/
│ ├── happy/
│ ├── sad/
│ ├── angry/
│ └── ...
└── test/
├── happy/
├── sad/
├── angry/
└── ...



## 🚀 Features

- Face emotion classification using CNN.
- Uses Keras layers such as Conv2D, MaxPooling2D, Dropout, and Dense.
- Image preprocessing and one-hot encoding of labels.
- Evaluation of model accuracy and loss.

## 🛠️ Technologies Used

- Python
- Keras / TensorFlow
- NumPy
- Pandas
- OpenCV / Image preprocessing tools

## 🧠 Model Architecture

The CNN model includes:
- Convolutional layers for feature extraction.
- MaxPooling layers for spatial downsampling.
- Dropout layers for regularization.
- Dense output layer for classification.

## 📊 Evaluation Metrics

- Accuracy
- Loss

## 📝 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yor user-name/face-emotion-detection.git
   cd face-emotion-detection

2.Install the required packages:
      pip install -r requirements.txt

3.Ensure the dataset is in the correct folder structure (images/train and images/test).

4.Run the Jupyter Notebook:
      jupyter notebook Face\ Emotion\ Detection.ipynb

### Future Improvements
-Add real-time webcam emotion detection using OpenCV.

-Deploy the model using a web app (e.g., Flask or Streamlit).

-Improve dataset quality and model accuracy

