import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained emotion detection model
try:
    model = load_model("facialemotionmodel.h5")  # Ensure the correct path to the model
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'facialemotionmodel.h5' file not found.")
    exit()

# Load Haar Cascade file for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the input image
def preprocess_face(image):
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (1 for grayscale)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image / 255.0  # Normalize pixel values

# Initialize webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

print("Starting real-time emotion detection...")
while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Unable to capture video from webcam.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_preprocessed = preprocess_face(face)

        # Predict the emotion
        pred = model.predict(face_preprocessed, verbose=0)
        emotion_label = labels[np.argmax(pred)]

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output in a window
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
