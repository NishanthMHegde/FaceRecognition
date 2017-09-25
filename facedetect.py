import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap =cv2.VideoCapture(0)
id = raw_input("Enter User ID")
sample=0
while True:
    ret,color = cap.read()
    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.3,5)
    for(x,y,w,h) in faces:
        sample =sample+1
        cv2.imwrite("dataset/User." + str(id)+"." + str(sample) + ".jpeg",img)
        cv2.rectangle(color,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.waitKey(100)
    cv2.imshow('img',color)
    cv2.waitKey(1)

    if(sample>20):
        break

cap.release()
cv2.destroyAllWindows()
