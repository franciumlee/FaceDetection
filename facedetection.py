import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('E:\python_code\FaceDetection\haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while True:
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(face_cascade)

        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=3, minSize=(10, 10))
        for (x, y, w, h) in facerect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = frame_gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

        k = cv2.waitKey(1)
        cv2.imshow('img', frame)
        if k==27:
            break
cv2.waitKey(0)
cv2.destroyAllWindows()