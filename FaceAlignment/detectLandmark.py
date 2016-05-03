# coding: utf-8
'''
reference: https://github.com/luoyetx/deep-landmark

Copyright (c) 2015, zhangjie
All rights reserved.
'''
import os
from os.path import join
import cv2
import caffe
import numpy as np
from config import FACE_ALIGNMENT_ROOT
from config import FACE_DETECTION_ROOT
from Common import BBox
from Common import Logger
import gc
import sys

class CNN(object):
    """
        Generalized CNN for simple run forward with given Model
    """

    def __init__(self, net, model):
        self.net = net
        self.model = model
        self.cnn = caffe.Net(net, model, caffe.TEST) # failed if not exists

    def forward(self, data, layer='fc2'):
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        # 2N --> Nx(2)
        t = lambda x: np.asarray([np.asarray([x[2*i], x[2*i+1]]) for i in range(len(x)/2)])
        result = t(result)
        return result

class Landmarker(object):
    """
        class Landmarker wrapper functions for predicting facial landmarks
    """

    def __init__(self):
        """
            Initialize Landmarker with files under VERSION
        """
        model_path = join(FACE_ALIGNMENT_ROOT, 'model')
        CNN_TYPES = ['LE1', 'RE1', 'N1', 'LM1', 'RM1', 'LE2', 'RE2', 'N2', 'LM2', 'RM2']
        level1 = [(join(model_path, '1_F.prototxt'), join(model_path, '1_F.caffemodel'))]
        level2 = [(join(model_path, '2_%s.prototxt'%name), join(model_path, '2_%s.caffemodel'%name)) \
                    for name in CNN_TYPES]
        level3 = [(join(model_path, '3_%s.prototxt'%name), join(model_path, '3_%s.caffemodel'%name)) \
                    for name in CNN_TYPES]
        self.level1 = [CNN(p, m) for p, m in level1]
        self.level2 = [CNN(p, m) for p, m in level2]
        self.level3 = [CNN(p, m) for p, m in level3]

    def detectLandmark(self, image, bbox, mode='fast'):
        """
            Predict landmarks for face with bbox in image
            fast mode will only apply level-1 and level-2
        """
        if not isinstance(bbox, BBox) or image is None:
            return None, False
        face = bbox.cropImage(image)
        face = cv2.resize(face, (39, 39)).reshape((1, 1, 39, 39))
        face = self._processImage(face)
        # level-1, only F in implemented
        landmark = self.level1[0].forward(face)
        # level-2
        landmark = self._level(image, bbox, landmark, self.level2, [0.16, 0.18])
        if mode == 'fast':
            return landmark, True
        landmark = self._level(image, bbox, landmark, self.level3, [0.11, 0.12])

    def _level(self, img, bbox, landmark, cnns, padding):
        """
            LEVEL-?
        """
        for i in range(5):
            x, y = landmark[i]
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[0])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d1 = cnns[i].forward(patch) # size = 1x2
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[1])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d2 = cnns[i+5].forward(patch)

            d1 = bbox.project(patch_bbox.reproject(d1[0]))
            d2 = bbox.project(patch_bbox.reproject(d2[0]))
            landmark[i] = (d1 + d2) / 2
        return landmark

    def _getPatch(self, img, bbox, point, padding):
        """
            Get a patch iamge around the given point in bbox with padding
            point: relative_point in [0, 1] in bbox
        """
        point_x = bbox.x + point[0] * bbox.w
        point_y = bbox.y + point[1] * bbox.h
        patch_left = point_x - bbox.w * padding
        patch_right = point_x + bbox.w * padding
        patch_top = point_y - bbox.h * padding
        patch_bottom = point_y + bbox.h * padding
        patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
        patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
        return patch, patch_bbox

    def _processImage(self, imgs):
        """
            process images before feeding to CNNs
            imgs: N x 1 x W x H
        """
        imgs = imgs.astype(np.float32)
        for i, img in enumerate(imgs):
            m = img.mean()
            s = img.std()
            imgs[i] = (img - m) / s
        return imgs

def drawLandmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 1, (0,255,0), -1)
    return img

def detectLandmarks(boundingboxList):
    """
        detect landmarks in `src` and store the result in `dst`
    """

    #bboxes = []
    #landmarks = []
    fl = Landmarker()
    logger = Logger(os.path.join(FACE_ALIGNMENT_ROOT, 'landmark.list'))

    # create bbox list
    fid = open(boundingboxList, 'r');
    fLines = fid.read().splitlines()
    fid.close()

    for line in fLines:
        word = line.split()
        filename = word[0]
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bbox = BBox([int(word[1]), int(word[2]), int(word[3]), int(word[4])])\
                .subBBox(0.1, 0.9, 0.2, 1)

        landmark, status = fl.detectLandmark(gray, bbox)

        '''
        get real landmark position
        '''
        landmark = bbox.reprojectLandmark(landmark)

        logger.writeMsg("%s" % filename)
        for x, y in landmark:
            logger.writeMsg(" %s %s" % (str(x), str(y)))
        logger.writeMsg('\n')

        '''
        free memory: force the Garbage Collector to release 
        '''
        gc.collect()

if __name__ == "__main__" :
    if len(sys.argv) != 2:
        print "Usage: python landmark.py boundingbox.list"
        sys.exit()

    detectLandmarks(sys.argv[1])

    print "LandmarkDetection Done!"
