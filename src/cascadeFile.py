# AJ Looney
# 9/7/15
# Facial-Detection
# CS160-2ATT

import os
import cv2


class CascadeFile:
    def __init__(self, filename):
        self.fileName = filename
        self.file = os.path.join(str(__file__).split("src")[0], "classifiers/" + self.fileName)
        assert os.path.isfile(self.file)

    def getClassifier(self):
        return cv2.CascadeClassifier(self.file)
