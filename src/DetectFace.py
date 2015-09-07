# AJ Looney
# 9/7/15
# Facial-Detection

import cv2
from cascadeFile import CascadeFile


class DetectFace:
    def __init__(self, vid, smile=False, eye=False):
        self.active = True
        self.smile = smile
        self.eye = eye
        self.vid = vid
        self.faceCascade = CascadeFile("haarcascade_frontalface_default.xml").getClassifier()

        if self.eye:
            pass  # self.eyeCascade = CascadeFile("haarcascade_eye.xml").getClassifier()
        if self.smile:
            self.smileCascade = CascadeFile("haarcascade_smile.xml").getClassifier()

        self.activate()

    def activate(self):
        while self.active:
            ret, frame = self.vid.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.4,
                minNeighbors=5,
                minSize=(15, 15)
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if self.smile:
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    smiles = self.smileCascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.05,
                        minNeighbors=5,
                        minSize=(15, 15)
                    )

                    for (ex, ey, ew, eh) in smiles[:1]:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)
                if self.eye:
                    pass  # Implement eye detection

            # Display the resulting frame
            cv2.imshow('Facial Detection', frame)

            k = cv2.waitKey(1)
            if k == 27 or k == 113:  # Esc key to stop
                self.active = False
                cv2.destroyAllWindows()
            if k == 115:  # Toggle smile
                self.smile = not self.smile
                continue
            elif k == -1:  # normally -1 returned,so don't print it
                continue
            else:
                continue
