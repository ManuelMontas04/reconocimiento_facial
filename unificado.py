import cv2
import numpy as np

cap = cv2.VideoCapture(0)


faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    # Captura un fotograma del video
    ret, frame = cap.read()
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

    cv2.imshow('frame', frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


