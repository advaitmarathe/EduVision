import numpy as np
import argparse
import imutils
import dlib
import cv2


pp = "shape_predictor_68_face_landmarks.dat"
def rect_to_box(rect):
    x = rect.left()
    y = rect.top()
    w= rect.right()
    h = rect.bottom()
    return (x,y,w,h)
def shape_to_np(shape,dtype="int"):
    coords = np.zeros((68,2),dtype=dtype)
    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pp)
image = cv2.imread("face1.jpg")
image = imutils.resize(image,width=500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Example",gray)
cv2.waitKey()
rects = detector(gray,1)
for (i,rect) in enumerate(rects):
    #make a region of interest out of your grayscale image and get the size of the rectangle
    #size of the rectangle should be checked to check whether its a good image to use
    #movement across frames, absolute location of eyebrows in these
    #check the rectangle if it has moved then you move everything else the same amount then analyze those(Difference between those images) 
    shape = predictor(gray,rect)
    shape = shape_to_np(shape)
    (x,y,w,h) = rect_to_box(rect)
    cv2.rectangle(image, (x,y),(w,h) , (0,255,0),2)
    for (x,y) in shape:
        cv2.circle(image, (x,y), 1, (0,0,255), -1)
cv2.imshow("output",image)
cv2.waitKey(0)
#work only using yourself as a model check which facial landmarks are moving and how
#this is limited to the resolution of the camera that you have
#check where it is in order