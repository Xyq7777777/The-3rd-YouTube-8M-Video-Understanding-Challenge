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

#### Patent Analysis

### System Design

<img src="https://github.com/Xyq7777777/The-3rd-YouTube-8M-Video-Understanding-Challenge/blob/CYW/%E6%9C%AA%E5%91%BD%E5%90%8D.jpg" />

### Major Components you think you will use

#### Technology Selection and reason behind selection including comparisons

1. Python
   
    We will use Python as our primary developing language and do the training part of the algotrithm. The reason using Python is that it is widely-accessible language and the starting code provided by the competition is also written in Python. In addition, some sample exercises uploaded by other groups are also written in Python. It will be convevient for us, especially not having any machine-learning background, to learn from.
   
2. Youtube API

    Youtube API will help us grab the existing videos with specific terms or labels on Youtube platform. It basically acts like Twitter API that we used for mini-project 1. We can, therefore, have some dataset to train and have some testing bench to verify.  Since this competition is Youtube-based, it should be better to grab the source directly from where analysis needed. 

3. YOLO frame-based object detection

    YOLO is a type of open-source algorithm that everyone can develope their methods based on this frame. This will help us to build a object-detection-machine-learning-based system. We chose it because it is open-source. 

#### Any test or verification programs

### Administrative

#### Project Lead: 
##### Sprint presentation for the class. 10/7 due.
##### Handover to Sprint 2 project lead
##### Sprint 2 Project Lead:

### Sprint 2 plan and assignments

