# AJ Looney
# 9/7/15
# Facial-Detection

import cv2


class VideoCapture:

    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(3, 420)
        self.video_capture.set(4, 340)
        self.video_capture.set(5, 60)

    def release(self):
        self.video_capture.release()

    def read(self):
        return self.video_capture.read()

