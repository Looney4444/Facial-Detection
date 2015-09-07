# AJ Looney
# 9/6/15
# FacialDetection

import os

import cv2

cascadePath_Face = os.path.join(str(__file__).split("src")[0], "classifiers/haarcascade_frontalface_default.xml")
cascadePath_Smile = os.path.join(str(__file__).split("src")[0], "classifiers/haarcascade_smile.xml")

assert os.path.isfile(cascadePath_Face)
assert os.path.isfile(cascadePath_Smile)

faceCascade = cv2.CascadeClassifier(cascadePath_Face)
smileCascade = cv2.CascadeClassifier(cascadePath_Smile)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 420)
video_capture.set(4, 340)
video_capture.set(5, 60)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(15, 15)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in smiles[:1]:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Facial Detection', frame)

    # When 'q' on the keyboard is hit the loop is exited
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
