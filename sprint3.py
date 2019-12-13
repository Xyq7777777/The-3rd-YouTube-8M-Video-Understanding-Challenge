# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages

import numpy as np
import argparse
import imutils
import time
import cv2
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox   


master = Tk() 
master.geometry('550x300')
master.title("Object Detection System")

data=("apple","aeroplane","backpack","banana","baseball bat","baseball glove","bear","bed","bench","bicycle","bird",
	"boat","book","bottle","bowl","broccoli","bus","cake","car","carrot","cat","cell phone","chair","clock","cow","cup",
	"diningtable","dog","donut","elephant","fire hydrant","fork","frisbee","giraffe","hair drier","handbag","horse","hot dog",
	"keyboard","kite","knife","laptop","microwave","motorbike","mouse","orange","oven","parking meter","person","pizza",
	"pottedplant","refrigerator","remote","sandwitch","scissors","sheep","sink","skateboard","skis","snowboard","sofa",
	"spoon","sports ball","stop sign","suitcase","surfboard","teddy bear","tennis racket","tie","toaster","toilet",
	"toothbrush","traffic light","train","truck","tvmonitor","umbrella","vase","wine glass","zebra")
e4 = Combobox(master,values=data)

Label(master, text='Input video name: ',font=("Arial Bold",16)).grid(row=0) 
Label(master, text='Output video name: ',font=("Arial Bold",16)).grid(row=1) 
Label(master, text='Path to YOLO: ',font=("Arial Bold",16)).grid(row=2)
Label(master, text='Keyword to search: ',font=("Arial Bold",16)).grid(row=3)
#lbl = Label(master, text='before')
#lbl.grid(row=4)
e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
#e4 = Entry(master) 
e1.grid(row=0, column=1) 
e2.grid(row=1, column=1) 
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
def clicked():
	ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--input", required=True,
	#	help="path to input video")
	#ap.add_argument("-o", "--output", required=True,
	#	help="path to output video")
	#ap.add_argument("-y", "--yolo", required=True,
	#	help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
		help="threshold when applyong non-maxima suppression")
	#ap.add_argument("-k", "--key", required=True, type=str, default = 'person',
	#        help="key to search")
	args = vars(ap.parse_args())

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([e3.get(), "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([e3.get(), "yolov3.weights"])
	configPath = os.path.sep.join([e3.get(), "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	# and determine only the *output* layer names that we need from YOLO
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream, pointer to output video file, and
	# frame dimensions
	vs = cv2.VideoCapture(e1.get())
	writer = None
	(W, H) = (None, None)
	st=[]
	# try to determine the total number of frames in the video file
	try:
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
			else cv2.CAP_PROP_FRAME_COUNT
		total = int(vs.get(prop))
		print("[INFO] {} total frames in video".format(total))

	# an error occurred while trying to determine the total
	# number of frames in the video file
	except:
		print("[INFO] could not determine # of frames in video")
		print("[INFO] no approx. completion time can be provided")
		total = -1
	index2 = 1
	cap = cv2.VideoCapture(e1.get())
	fps = cap.get(cv2.CAP_PROP_FPS)
	duration = total/ fps
	# loop over frames from the video file stream
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		index2 = index2 + 1
		if not grabbed:
			break

		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
			



				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args["confidence"]:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
		
					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)




	        
		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])
		

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				if LABELS[classIDs[i]] == e4.get():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the frame
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]],
						confidences[i])
					cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
					if index2 not in st:
						st.append(index2)

		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(e2.get(), fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)

			# some information on processing single frame
			if total > 0:
				elap = (end - start)
				print("[INFO] single frame took {:.4f} seconds".format(elap))
				print("[INFO] estimated total time to finish: {:.4f}".format(
					elap * total))

		# write the output frame to disk
		writer.write(frame)
	st2=[]
	st3=[]
	for index2 in st:
		st2.append(int(index2*(duration/ total)))
	for k in st2:
		if k not in st3:
			st3.append(k)
	fo = open("cut.txt", "w")
	fo.write(e2.get()+'\n')
	for item in st3:
		fo.write(str(item)+' ')
	fo.close

	messagebox.showinfo('Results',"The item you want is at(s):\n {}".format(st3))
	print("[INFO] cleaning up...")
	writer.release()
	vs.release()
    #res = 'after:' + e1.get()
    #lbl.config(text = res)
btn = Button(master, text="Go", bg="red",font=("Arial Bold",20), command=clicked)
 
btn.grid(row=5, column=2)
mainloop()

file=open("cut.txt", "r")
name=file.readline() 
name=name[:-1] 
st6=file.readline()
st4=st6.split(' ')
del st4[-1]
st5=list(map(int,st4))


z=0
start=[]
end=[]
start.append(st5[0])
length_st5=len(st5)
while z<length_st5-1:
	if (st5[z+1]-st5[z]==1):
		z+=1
	else: 
		end.append(st5[z])
		start.append(st5[z+1])
		z+=1
end.append(st5[z])
#print(start)
#print(end)
for ti in range(len(start)):
	if (start[ti] != end[ti]):
		ffmpeg_extract_subclip(name, start[ti], end[ti], targetname=str(ti)+name)
	else:
		ffmpeg_extract_subclip(name, start[ti], end[ti]+1, targetname=str(ti)+name)


for lebron in range (len(start)):
	cap = cv2.VideoCapture(str(lebron)+name)
	 
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
	 
	# Read until video is completed
	while(cap.isOpened()):
	  # Capture frame-by-frame
	  ret, frame = cap.read()
	  if ret == True:
	 
	    # Display the resulting frame
	    cv2.imshow('Frame',frame)
	 
	    # Press Q on keyboard to  exit
	    if cv2.waitKey(25) & 0xFF == ord('q'):
	      break
	 
	  # Break the loop
	  else: 
	    break
	 
	# When everything done, release the video capture object
	cap.release()
	 
	# Closes all the frames
	cv2.destroyAllWindows()