import cv2 as cv
import numpy as numpy
from matplotlib import pyplot as plt

img = cv.imread('images/12_test.tif',cv.IMREAD_UNCHANGED)
#img = cv.GaussianBlur(img,(5,5),0)
#th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\cv.THRESH_BINARY,11,2)
#im2, contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
g=img[:,:,2]
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(g)
median = cv.medianBlur(cl1,5)
for i in range(100,230):
    ret,th2=cv.threshold(median,i,255,cv.THRESH_BINARY)
    median = cv.medianBlur(th2,5)
    contours, hierarchy = cv.findContours(median, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours[1], -1, (0,255,0), 3)
cv.imshow("test",img)
cv.waitKey(-1)