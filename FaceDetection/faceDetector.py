# -*- coding: utf-8 -*-
from Common import BBox
from Common import Logger
from config import FACE_DETECTION_ROOT
import cv2
import os
import sys

class FaceDetector(object):

    def __init__(self):
        self.dataSetFile = ''
        self.successLogger = Logger(os.path.join(FACE_DETECTION_ROOT, 'faceDetection.success'))
        self.errorLogger = Logger(os.path.join(FACE_DETECTION_ROOT, 'faceDetection.error'))
        self.boundingboxFile = Logger(os.path.join(FACE_DETECTION_ROOT, 'boundingbox.list'))

        self.cc = cv2.CascadeClassifier(os.path.join(FACE_DETECTION_ROOT, 'haarcascade_frontalface_alt.xml'))

    def setDataSetFile(self, filePath):
        self.dataSetFile = filePath

    def run(self):
        fid = open(self.dataSetFile, 'r').read().splitlines()

        for src in fid:
            self._detectFace(src)

    def _detectFace(self, imgPath):
        img = cv2.imread(imgPath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = self.cc.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, \
                    minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

        if not len(rects):
            self.errorLogger.writeMsg("%s\n" % imgPath)
            return

        for rect in rects:
            rect[2:] += rect[:2]

            self.successLogger.writeMsg("%s\n" % imgPath)

            '''
            boundingbox format: left right top bottom
            '''
            self.boundingboxFile.writeMsg("%s %s %s %s %s\n" % \
                (imgPath, str(rect[0]), str(rect[2]), str(rect[1]), str(rect[3])))

if __name__ == "__main__" :
    if len(sys.argv) != 2:
        print "Usage: python faceDetector.py imageList.txt"
        sys.exit()

    fd = FaceDetector()
    fd.setDataSetFile(sys.argv[1])
    fd.run()

    print "FaceDetection Done!"
