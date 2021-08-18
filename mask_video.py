# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

def match():
	original = cv2.imread("data/original.JPG")
	original = cv2.resize(original, (200,200))
	# convert the images to grayscale
	original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	# In[5]:
	dataset= pd.read_csv('data.csv',sep=',')
	data = dataset.iloc[:, :]
	data
	x = data.iloc[:, :-1].values 
	d = dataset.iloc[:, 2]
	#print(d[2])
	# In[6]:
	values = [];
	for i in range(0,len(d)):
		image = cv2.imread(str(d[i]))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		values.append(compare_images(original, image, "Original vs."+str(d[i])) )
	min=values[0]
	for i in range(1,len(d)):
		if min>values[i]:
			min=i 
	print(x[min])
	fig = plt.figure("Match")
	#plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	#plt.suptitle(x[min]) 
	plt.suptitle(str(x[min]))
#print(title);
# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(original, cmap = plt.cm.gray)
	plt.axis("off")
# show the second image
	ax = fig.add_subplot(1, 2, 2)
	image = cv2.imread(str(d[min]))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the images
	plt.show()
	return str(x[min])



# In[2]:


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# In[3]:


def compare_images(imageA, imageB, title):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # setup the figure
   # fig = plt.figure(title)
    #plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    #print(title);
    # show first image
   # ax = fig.add_subplot(1, 2, 1)
    #plt.imshow(imageA, cmap = plt.cm.gray)
    #plt.axis("off")
    # show the second image
    #ax = fig.add_subplot(1, 2, 2)
    #plt.imshow(imageB, cmap = plt.cm.gray)
    #plt.axis("off")
    # show the images
   # plt.show()
    return m


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture("C:\\Users\\Neha\\Documents\\Zoom\\2021-05-10 12.24.19 Neha Bheemisetty 19BCE1123's Zoom Meeting 73895884754\\zoom_3.MP4")
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	i=0
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		capture = 1 if label=="Mask" else 0
			
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


	# show the output frame
	cv2.imshow("Frame", frame)
	if capture== 0 :
		print("The details of the person who is not wearing the mask are:")
		name = './data/original'+'.JPG'
		i+=1 
		cv2.imwrite(name,frame)
		person=match()
		print(person)
	'''	# In[4]:
		# load the images -- the original, the original + contrast,
		# and the original + photoshop
		original = cv2.imread("data/original.JPG")
		original = cv2.resize(original, (200,200))
		# convert the images to grayscale
		original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
		# In[5]:
		dataset= pd.read_csv('data.csv',sep=',')
		data = dataset.iloc[:, :]
		data
		x = data.iloc[:, :-1].values 
		d = dataset.iloc[:, 2]
		print(d[2])
		# In[6]:
		values = [];
		for i in range(0,len(d)):
			image = cv2.imread(str(d[i]))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			values.append(compare_images(original, image, "Original vs."+str(d[i])) )
		min=values[0]
		for i in range(1,len(d)):
			if min>values[i]:
				min=i 
		print(x[min])
		fig = plt.figure("Match")
		#plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
		#plt.suptitle(x[min]) 
		plt.suptitle(str(x[min]))
#print(title);
# show first image
		ax = fig.add_subplot(1, 2, 1)
		plt.imshow(original, cmap = plt.cm.gray)
		plt.axis("off")
# show the second image
		ax = fig.add_subplot(1, 2, 2)
		image = cv2.imread(str(d[min]))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		plt.imshow(image, cmap = plt.cm.gray)
		plt.axis("off")
# show the images
		plt.show()
#print(dict[min]);'''

	#if(label=="No Mask"):
	#	cv2.imwrite("C:\Users\Neha\Desktop\Face-Mask-Detection-master\original.JPG")
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


