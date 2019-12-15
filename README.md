# The-3rd-YouTube-8M-Video-Understanding-Challenge
## Sprint 1

### Product Definition
#### Product Mission

This product aims to help users search for certain elements within a video quickly, efficiently and correctly. The topic, hashtags, subtitles are the literal keys for searching while the images, screeshots may also contain desired results in graphical format. Digging out the information stored in a video which has no literal description to look up is our primary goal. Once the product is completed, the results after searching is going to be more related to the keywords users entered with the timing indicator installed.

#### Target User(s)

General public, researchers, merchants.

#### User Stories

I, the general public, would like to have more accurate searching results even in non-literal format.</br>
I, the general public, would like to have timing indicator jumps to the direct results I want in a video.</br>
I, the researchers, would like to collect the searching ddata and then analyze the which part of the video is more valuable to users in general.</br>
I, the merchants, would like to know when to advertise my product in the video based on collected data.</br>
I, the researcher, should be able to get a result with at least 70% accuracy from this model.</br>
I, the researcher, should be able to receive the source code of this model so that I can improve it.</br>
I, the general public, should be able to know the duration of the moments I want in the video.</br>
I, the merchants and researchers, would like to know the most popular keywords for searching recently.

#### MVP

Find out the information stored in non-literal format such as videos or photos which may be related to the keywords provided by users.

#### User Interface Design for main user story if required

It should have a place for users to enter the keywords and output the results containing the timing indicator for every possibly related video listed.

### Product Survey

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

#### Any test or verification programs

We are still learining how to detect real time objects.
[https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/]

## Sprint 2

### Basic Detection

   In this stage, we tried to run the basic YOLO algorithm on certain videos. With the original design, the objects within a video can all be identified and labeled. The bounding box contains the label name and the confidence of that specific detection.
   
<所有label都有顯示的截圖>

### Designated Object Detection

   After having the knowledge on how the algorithm process, we would like to further the ability of detecting only one object which is designated by users' input. We eliminated the bounding box of non-keyword labels, leaving the only desire label output be presented in the generated video after processing.
   
<只有一個label的截圖>

### Timing Return

   Having the ability to identify keyword-only objects, we would like to mark down the specific timing and return to users. In this part, we had the main python script store a list containing all timing and then print the list onto a text file. The ffmpeg library is , thereby, included. The algorithm can actuall identify timing in millisecond scale (since we are counting by the frame number), but we made it round to second scale because of the practical use. For the future step, pop out the desired sections, we may use this text file to help our algorithm fulfill the goal.    

<顯示timing的txt file截圖>

## Sprint 3

### User Interface

   Since we need the inputs from users, we may limit the input options to the model we provided. Otherwise, the input may go beyond the label we can do the service for them. We implemented this by tk which is a supported library for Python. In "coco", we support 80 labels and we, therefor, make a scroll box for keyword input so that there will not be any invalid inputs. The input file should be under the same folder as our script and model. The path field is designed for detection models which leave a possible development that users/ developers may want to import their own models with higher accuracy or more keywords(labels) supported in future. 

<UI的截圖,有能下拉的>

### Pop out Labeled Sections 

   We had the script read the timing text file (in sprint 2) after processing to know which sections to show automatically. Therefore, users should see the sections of desired keyword directly after closing the UI window and the whole processed video can also be accessed manually within the same folder.
   
<跑完之後Popout的截圖>

### Training Method

   We used the coco dataset which contains thousands of images to train. In addition, darknet files are used in the process to help us set the parameters such as image size, training rate, learning rate, image angles and so on. It also has a basic sample script of grabbing frames in a video for us to develop on.
   
<training的terminal截圖>

## Reference
1. YOLO website with different generations and link to coco dataset https://pjreddie.com/darknet/yolo/ </br>
2. Tutorial on training https://www.learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/ </br>
3. Basic YOLO algorithm detecting all labels https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/ </br>
4. OpenCV tutorial https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html </br>
5. Counting time on videos by frames https://stackoverflow.com/questions/2017843/fetch-frame-count-with-ffmpeg </br>

## Work Distribution
Sprint1: Building basical object detection algorithm: Lingtao Jiang & Yi-Wei Chen & Yaqun Xia

Sprint2: Improving algorithm to detect specific object and implement returning the time of the detected object in the video:
Yi-Wei Chen & Lingtao Jiang

Sprint3: Makeing the user interface and implementing poping out the labeled sections of the detected video: Lingtao Jiang & Yi-Wei Chen

Attempt to training the coco dataset: Lingtao Jiang & Yi-Wei Chen & Yaqun Xia

