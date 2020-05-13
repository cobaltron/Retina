import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from imutils import contours
from skimage import measure
import imutils

img = cv.imread('images/12_test.tif',cv.IMREAD_UNCHANGED)
img = cv.GaussianBlur(img,(5,5),0)
#th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\cv.THRESH_BINARY,11,2)
#im2, contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
g=img[:,:,1]
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(g)
blurred = cv.GaussianBlur(cl1, (11, 11), 0)
ret,th2=cv.threshold(blurred,200,255,cv.THRESH_BINARY)
thresh = cv.erode(th2, None, iterations=2)
thresh = cv.dilate(th2, None, iterations=4)
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
 
# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
 
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv.countNonZero(labelMask)
 
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 300:
		mask = cv.add(mask, labelMask)
cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#cnts = contours.sort_contours(cnts)[0]
 
# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv.boundingRect(c)
	((cX, cY), radius) = cv.minEnclosingCircle(c)
	cv.circle(img, (int(cX), int(cY)), int(radius),
		(0, 0, 255), 3)
 
# show the output image
cv.imshow("test",img)
cv.waitKey(-1)
