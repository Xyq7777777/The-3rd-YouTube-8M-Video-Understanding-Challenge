# The-3rd-YouTube-8M-Video-Understanding-Challenge
## Sprint 1

### Product Definition
#### Product Mission

This product aims to help users search for certain elements within a video quickly, efficiently and correctly. The topic, hashtags, subtitles are the literal keys for searching while the images, screeshots may also contain desired results in graphical format. Digging out the information stored in a video which has no literal description to look up is our primary goal. Once the product is completed, the results after searching is going to be more related to the keywords users entered with the timing indicator installed.

#### Target User(s)

General public, researchers, merchants.

#### MVP

Find out the information stored in non-literal format such as videos or photos which may be related to the keywords provided by users.

#### Existing similar products

The YOLO detection system: It's easy to process images, it runs a convolutional network on the image and thresholds the resulting detections by the module's confidence. It can predict what objects are present and where they are. Benefits: extremely fast, high precision, reasons globally about the image when making predictions.

#### Patent Analysis

1. CN107247956A-Fast target detection method based on grid judgment 

   The invention discloses a fast target detection method based on grid judgment. The method comprises the following steps:  
S1. Gridding an image.  
S2. Extracting the features of the grid areas.  
S3. Judging and merging the grids: first, judging whether each grid belongs to a specified target object according to a            regression model trained in advance, and then, merging the grids into an initial object window according to the object category to which each grid belongs.  
S4. Carrying out bounding-box regression on the initial object window through a bounding-box regression method. The accuracy and speed of target detection can both be ensured.  

2. CN106886795A-Object identification method based on substantial object in image 

   The invention relates to an object identification method based on a substantial object in an image. The method comprises that in a training process, a classification database which comprises first characteristic vectors for describing objects is established; and in an identification process, a picture including objects is input to a deep convolutional neural network, and divided into M*M grids, each grid predicts N candidate frames to obtain the probability that objects exist in the frame, when the object probability is greater than or equivalent to a predetermined threshold, the candidate frame is selected as a first effective candidate frame, an image of the first effective candidate frame is input to a classified neural network to obtain a second characteristic vector, and a k nearest neighbor (KNN) classification algorithm is executed to obtain a class of the object on the basis of the second and first characteristic vectors and the classification database.

### System Design

![Image text](https://github.com/Xyq7777777/The-3rd-YouTube-8M-Video-Understanding-Challenge/blob/master/sys.png)

### Major Components you think you will use

#### Technology Selection and reason behind selection including comparisons

1. Python
   
    We will use Python as our primary developing language and do the training part of the algotrithm. The reason using Python is that it is widely-accessible language and the starting code provided by the competition is also written in Python. In addition, some sample exercises uploaded by other groups are also written in Python. It will be convevient for us, especially not having any machine-learning background, to learn from.
   
2. OpenCV

    OpenCV is a library contains lots of functions related to image processing and video manipulation. We will include these functions to help us deal with the videos consisting of frames.

3. YOLO frame-based object detection

    YOLO is a type of open-source algorithm that everyone can develope their methods based on this frame. This will help us to build a object-detection-machine-learning-based system. We chose it because it is open-source. 

#### Test or verification programs

We are still learining how to detect real time objects.
[https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/]



## Sprint 3 
For sprint, I was trying to do the model training. I have tried several method which two of them are executable.

### 1. YOLO offical website  
https://pjreddie.com/darknet/yolo/  

#### Steps I followed
I followed the Training YOLO on COCO part:  
- Download the darknet 
<img src= "https://github.com/Xyq7777777/The-3rd-YouTube-8M-Video-Understanding-Challenge/blob/XYQ/darknet.jpg">
- Download the COCO data
<img src= "https://github.com/Xyq7777777/The-3rd-YouTube-8M-Video-Understanding-Challenge/blob/XYQ/download.jpg">
- Change the parameter     
- Train the model  
<img src= "https://github.com/Xyq7777777/The-3rd-YouTube-8M-Video-Understanding-Challenge/blob/XYQ/training process.jpg">

#### Conclusion  
I have met a few errors:
- Darknet error: Solved by search the new darnet53.conv.74
<img src= "https://github.com/Xyq7777777/The-3rd-YouTube-8M-Video-Understanding-Challenge/blob/XYQ/darkneterror.jpg">

- Result error: Still cannot be solved by download the new file
<img src= "https://github.com/Xyq7777777/The-3rd-YouTube-8M-Video-Understanding-Challenge/blob/XYQ/result.jpg">


### 2. Yolo-v3 and Yolo-v2 for Windows and Linux  
This is the improved version for the offical website  
The github link is:  
https://github.com/AlexeyAB/darknet  

#### Steps I followed:  

How to train (to detect your custom objects):
(to train old Yolo v2 yolov2-voc.cfg, yolov2-tiny-voc.cfg, yolo-voc.cfg, yolo-voc.2.0.cfg, ... click by the link)

Training Yolo v3:

Create file yolo-obj.cfg with the same content as in yolov3.cfg (or copy yolov3.cfg to yolo-obj.cfg) and:
change line batch to batch=64
change line subdivisions to subdivisions=8
change line max_batches to (classes*2000 but not less than 4000), f.e. max_batches=6000 if you train for 3 classes
change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
change line classes=80 to your number of objects in each of 3 [yolo]-layers:
https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610
https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696
https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783
change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer
https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603
https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689
https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776
when using [Gaussian_yolo] layers, change [filters=57] filters=(classes + 9)x3 in the 3 [convolutional] before each [Gaussian_yolo] layer
https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L604
https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L696
https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L789
So if classes=1 then should be filters=18. If classes=2 then write filters=21.

(Do not write in the cfg-file: filters=(classes + 5)x3)

(Generally filters depends on the classes, coords and number of masks, i.e. filters=(classes + coords + 1)*<number of mask>, where mask is indices of anchors. If mask is absence, then filters=(classes + coords + 1)*num)

So for example, for 2 objects, your file yolo-obj.cfg should differ from yolov3.cfg in such lines in each of 3 [yolo]-layers:

[convolutional]
filters=21

[region]
classes=2
Create file obj.names in the directory build\darknet\x64\data\, with objects names - each in new line

Create file obj.data in the directory build\darknet\x64\data\, containing (where classes = number of objects):

classes= 2
train  = data/train.txt
valid  = data/test.txt
names = data/obj.names
backup = backup/
Put image-files (.jpg) of your objects in the directory build\darknet\x64\data\obj\

You should label each object on images from your dataset. Use this visual GUI-software for marking bounded boxes of objects and generating annotation files for Yolo v2 & v3: https://github.com/AlexeyAB/Yolo_mark

It will create .txt-file for each .jpg-image-file - in the same directory and with the same name, but with .txt-extension, and put to file: object number and object coordinates on this image, for each object in new line:

<object-class> <x_center> <y_center> <width> <height>

Where:

<object-class> - integer object number from 0 to (classes-1)
<x_center> <y_center> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
atention: <x_center> <y_center> - are center of rectangle (are not top-left corner)
For example for img1.jpg you will be created img1.txt containing:

1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
Create file train.txt in directory build\darknet\x64\data\, with filenames of your images, each filename in new line, with path relative to darknet.exe, for example containing:
data/obj/img1.jpg
data/obj/img2.jpg
data/obj/img3.jpg
Download pre-trained weights for the convolutional layers (154 MB): https://pjreddie.com/media/files/darknet53.conv.74 and put to the directory build\darknet\x64

Start training by using the command line: darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74

To train on Linux use command: ./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74 (just use ./darknet instead of darknet.exe)

(file yolo-obj_last.weights will be saved to the build\darknet\x64\backup\ for each 100 iterations)
(file yolo-obj_xxxx.weights will be saved to the build\darknet\x64\backup\ for each 1000 iterations)
(to disable Loss-Window use darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show, if you train on computer without monitor like a cloud Amazon EC2)
(to see the mAP & Loss-chart during training on remote server without GUI, use command darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map then open URL http://ip-address:8090 in Chrome/Firefox browser)
8.1. For training with mAP (mean average precisions) calculation for each 4 Epochs (set valid=valid.txt or train.txt in obj.data file) and run: darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map

After training is complete - get result yolo-obj_final.weights from path build\darknet\x64\backup\
After each 100 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just start training using: darknet.exe detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights

(in the original repository https://github.com/pjreddie/darknet the weights-file is saved only once every 10 000 iterations if(iterations > 1000))

Also you can get result earlier than all 45000 iterations.

Note: If during training you see nan values for avg (loss) field - then training goes wrong, but if nan is in some other lines - then training goes well.

Note: If you changed width= or height= in your cfg-file, then new width and height must be divisible by 32.

Note: After training use such command for detection: darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights

Note: if error Out of memory occurs then in .cfg-file you should increase subdivisions=16, 32 or 64: link

#### Conclusion
All steps passed but some new errors apperars after using the command lines, which cannot solved in a short time.
<img src= "https://github.com/Xyq7777777/The-3rd-YouTube-8M-Video-Understanding-Challenge/blob/XYQ/newerror.jpg">
