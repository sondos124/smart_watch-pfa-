import cv2
import numpy as np
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# Configuration for image processing
folder = "/home/nihed/VitalTrack/computer_vision/data/Angry"  # Absolute path to the "Happy" folder
  # Folder to save captured images
counter = 0  # Counter to track the number of saved images

# Main loop for continuous image processing
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the region of interest (ROI) containing the face
        face_roi = img[y:y+h, x:x+w]

        # Save the captured image if 's' key is pressed
        key = cv2.waitKey(1)
        if key == ord('s'):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', face_roi)
            print(counter)

    # Display the processed image with face detection
    cv2.imshow("Face Detection", img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()