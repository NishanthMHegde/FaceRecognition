import numpy as np
import cv2

img = cv2.imread("image.jpg")
retval,threshold = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval2,threshold2 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,11)

cv2.imshow('gaus',gaus)
cv2.imshow('gray',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
