import cv2

# Load the pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Webcam
cap = cv2.VideoCapture(0)

# Loop through each frame of the webcam feed
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(frame)

    # Draw a rectangle around each detected face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # Display the frame with the detected faces
    cv2.imshow('frame',frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
