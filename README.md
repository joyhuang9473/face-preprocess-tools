Face-preprocess-tools
=========

Face-preprocess-tools is a collection of tools for preprocessing face images. Images will go through `FaceDetection`, `FaceAlignment`, `FaceCropping` processes.

BaseLine
-----------------------

`FaceDetection` -> `FaceAlignment` -> `FaceCropping`

FaceDetection
-----------------------

### Haar Feature-based Cascade Classifiers ###

@param:

- `imageList.txt`: a list file, which contains image file paths.

@output:

- `faceDetection.success:` a list file, which records successful detection.
- `faceDetection.error`: a list file, which records error detection.
- `boundingbox.list`: a list file, which contains bounding box information of successful detection images.

```
$ cat imageList.txt
/workspace/datasets/CASIA-WebFace/5587543/001.jpg
/workspace/datasets/CASIA-WebFace/5587543/002.jpg
...

$ python FaceDetection/faceDetector.py imageList.txt
FaceDetection Done!
```

FaceAlignment
-----------------------

### Deep Convolutional Network Cascade for Facial Point Detection ###

Two parts:

1. Landmark Detection
2. Face Alignment

**Landmark Detection**

@param:

- `boundingbox.list`: the output file `boundingbox.list` of FaceDetection process.

@output:

- `landmark.list`: a list file, which contains five facial points of face images listed in `boundingbox.list`.

```
$cat boundingbox.list
/workspace/datasets/CASIA-WebFace/5587543/001.jpg 47 203 46 202
/workspace/datasets/CASIA-WebFace/5587543/002.jpg 39 195 51 207
...

$ python FaceAlignment/detectLandmark.py boundingbox.list
LandmarkDetection Done!
```

**Face Alignment**

@param:

- `landmark.list`: the output file `landmark.list` of Landmark Detection process.
- `PATH_Aligned_image_dir`: the destination directory path of aligned output images.

@output:

- `faceAlignemnt.list`: a list file, which records successful alignment.
- `Aligned_image_dir`: a directory, which contains the aligned output images listed in `faceAlignemnt.list`.

```
$ cat landmark.list
/workspace/datasets/CASIA-WebFace/5587543/001.jpg 98.1535536766 111.477871895 155.723092651 106.953521013 130.264041901 137.168443489 103.697720623 165.793231583 153.124426651 165.626620483
/workspace/datasets/CASIA-WebFace/5587543/002.jpg 80.4625348091 113.501712513 137.962782669 107.797487926 101.989186478 146.925302696 89.6215407372 172.050915527 136.488462639 161.69779644
...

$ python FaceAlignment/alignment.py landmark.list CASIA-WebFace-Aligned
FaceAlignment Done!
```

FaceCropping
-----------------------

@param:

- `faceAlignment.list`: the output file `faceAlignment.list` of Face Alignment process.
- `PATH_Cropped_image_dir`: the destination directory path of cropped output images.
- `dst_width`: the width of cropped output images.
- `dst_heigh`: the height of cropped output images.

@output:

- `Cropped_image_dir`: a directory, which contains the cropped output images listed in `faceAlignemnt.list`.

```
$ cat faceAlignment.list
/workspace/datasets/CASIA-WebFace/5587543/001.jpg
/workspace/datasets/CASIA-WebFace/5587543/002.jpg
...

$ python FaceCropping/facecrop.py faceAlignment.list CASIA-WebFace-Cropped 62 62
FaceCropping Done!
```

Reference
-----------------------

[1]. zhangjie, [luoyetx/deep-landmark](https://github.com/luoyetx/deep-landmark), 2016-01-26

[2]. RiweiChen, [RiweiChen/FaceTools](https://github.com/RiweiChen/FaceTools), 2016-03-11
