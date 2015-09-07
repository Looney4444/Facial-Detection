# AJ Looney
# 9/6/15
# FacialDetection

import cv2

cascadePath_Face = './classifiers/haarcascade_frontalface_default.xml'
cascadePath_Smile = './classifiers/haarcascade_smile.xml'

faceCascade = cv2.CascadeClassifier(cascadePath_Face)
smileCascade = cv2.CascadeClassifier(cascadePath_Smile)


video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
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
            scaleFactor=1.1,
            minNeighbors=600,
            minSize=(10, 10)
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
