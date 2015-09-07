# AJ Looney
# 9/6/15
# FacialDetection

from videoCapture import VideoCapture
from detectFace import DetectFace


def main():
    vid = VideoCapture()

    face = DetectFace(vid, smile=False, eye=False)
    face.activate()

    vid.release()
if __name__ == '__main__':
    main()

