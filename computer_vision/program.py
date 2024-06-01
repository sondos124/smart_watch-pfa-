import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import requests

# Load the pre-trained Keras model for emotion detection
model_path = "/home/nihed/VitalTrack/computer_vision/Model/keras_model.h5"
model = load_model(model_path)

# Initialize the video capture from the default camera (0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Initialize the face cascade classifier
face_cascade_path = "/home/nihed/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Define emotion labels
emotion_labels = {0: "Happy", 1: "Angry", 2: "Sad"}

# Main loop for continuous image processing
while True:
    # Read a frame from the camera
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f"Detected faces: {len(faces)}")

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Crop the region of interest (ROI) containing the face
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face ROI to match the input size of the model
        face_img = cv2.resize(face_roi, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        # Perform emotion detection
        emotion_probabilities = model.predict(face_img)[0]
        predicted_emotion = np.argmax(emotion_probabilities)
        emotion_label = emotion_labels[predicted_emotion]
        print(f"Detected emotion: {emotion_label}")

        # Send the detected emotion to the Flask server
        try:
            response = requests.post('http://127.0.0.1:5001/update_emotion', json={'emotion': emotion_label})
            print(f"Sent emotion data, response status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print("Error sending emotion data:", e)

        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the detected emotion label above the face
        cv2.putText(img, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the processed image with face detection
    cv2.imshow("Emotion Detection", img)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
