import numpy as np
import argparse
import imutils
import dlib
import cv2
import copy
pp = "shape_predictor_68_face_landmarks.dat"


eyebrows = {
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
"jaw": (0, 17),
"right_eye": (36, 42),
    "left_eye": (42, 48)
}

eyesdistance = {
    "lefteye1": 0,
    "lefteye2": 0,
    "righteye1": 0,
    "righteye2": 0
}


def distance(x1,y1,x2,y2):
    y = y2-y1
    x = x2-x1
    return pow((pow(y,2) + pow(x,2)), 0.5)

def distance2(a,b):
    y = a[1]- b[1]
    x = a[0]-b[0]
    return pow((pow(y, 2) + pow(x, 2)), 0.5)
def rect_to_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()
    h = rect.bottom()
    return (x, y, w, h)
def EAR(values):
    v1 = distance2(values[1],values[5])
    v2 = distance2(values[2],values[4])
    v3 = 2 * (distance2(values[0],values[3]))
    return ((v1 +v2)/v3)
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pp)
distances2 = {
    "eyebrow1" : 10,
    "eyebrow2": 10 ,
    "eyebrow3": 10,
    "eyebrow4": 10,
    "eyebrow5": 10,
    "eyebrow6": 10,
    "eyebrow7": 10,
    "eyebrow8": 10,
    "eyebrow9": 10,
    "eyebrow10": 10
}
final_face = [0,1,2,3,4,5]

for i in range(20):
    ret1, image1 = cap.read(0)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Example", gray)
    # cv2.waitKey()

    faces = detector(gray, 1) #facial region
    final_faces = faces
    eye_points = []
    for (i, rect) in enumerate(faces): #looping through the information
        shape = predictor(gray, rect)#facial points
        shape = shape_to_np(shape) #array that produces x,y values for each numbered point
        (x, y, w, h) = rect_to_box(rect) #draws bounding box
        cv2.rectangle(image1, (x, y), (w, h), (0, 255, 0), 2)
        for i in range(1,6):
            distances2["eyebrow"+str(i)] = distance(shape[2+i][0], shape[2+i][1], shape[16+i][0], shape[16+i][1])
            cv2.line(image1,(shape[2+i][0],shape[2+i][1]),(shape[16+i][0], shape[16+i][1]),(255,0,0),2)
            distances2["eyebrow"+str(11-i)] = distance(shape[14-i][0], shape[14-i][1], shape[27-i][0], shape[27-i][1])
            cv2.line(image1,(shape[14-i][0], shape[14-i][1]),(shape[27-i][0], shape[27-i][1]),(255,0,0),2)

        for i in range(0,6):
            eye_points.append(shape[36+i])
            for (name, (i, j)) in eyebrows.items():
                for (x, y) in shape[i:j]:
                    cv2.circle(image1, (x, y), 1, (0, 255, 255), -1)
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        cv2.imshow("Image", image1)

        if (len(eyesdistance)>0 and len(distances2)>0 and len(eye_points)>0):
            final_face[0]=(copy.deepcopy(eyesdistance)) #0
            final_face[1]=(copy.deepcopy(distances2)) #1
            final_face[2]=(copy.deepcopy(eye_points)) #2
            final_face[3] =(0) #values of the EAR #3
            final_face[4] =(0) #Counter #4
            final_face[5] =(0) #Score Value #5
            break
        cv2.imshow("Image", image1)
cap.release()
print(final_face)


cap = cv2.VideoCapture(0)

while True:
    difference_level = 10
    ret, image = cap.read()
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Example", gray)
    # cv2.waitKey()
    rects = detector(gray, 1) #facial region
    #if you want to check when a face leaves go back to Eyebrow_video (Match every face with the one inside and take them out of the array while saving their index
    for (i, rect) in enumerate(rects): #looping through the information
        shape = predictor(gray, rect)#facial points
        shape = shape_to_np(shape) #array that produces x,y values for each numbered point
        (x, y, w, h) = rect_to_box(rect) #draws bounding box
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

        face_number = 100;
        eye_points = []
        for i in range(0,6):
            eye_points.append(shape[36+i])
        EARval = EAR(eye_points)
        #Need to save this ear value into the next one and then match if they are both less than <0 then we increase the counter
        print(EARval)
        for i in range(1,6):

            cv2.line(image,(shape[2+i][0],shape[2+i][1]),(shape[16+i][0], shape[16+i][1]),(255,0,0),2)
            cv2.line(image,(shape[14-i][0], shape[14-i][1]),(shape[27-i][0], shape[27-i][1]),(255,0,0),2)
        for i in range(1, 3):

            cv2.line(image, (shape[36 + i][0], shape[36 + i][1]), (shape[42 - i][0],
                                                      shape[42 - i][1]), (0, 255, 0), 2)

            cv2.line(image, (shape[42 + i][0], shape[42+i][1]), (shape[48 - i][0],
                                                           shape[48 - i][1]), (0, 255, 0), 2)
        #final_Face = eyedistance, eyebrows, counter, score
        if (EARval<0.19 and final_face[3]>0.19):
            final_face[4] = 0
        if(EARval<0.19 and final_face[4] == 0):
            final_face[4] = 1
            final_face[3] = EARval
        if(EARval<0.19 and final_face[3]<0.19):
            final_face[4] = final_face[4] + 1
            final_face[3] = EARval
        if (final_face[4] == 100):
            final_face[5] = final_face[5]+100
            final_face[4] = 0
        print("This is the counter" + str(final_face[4]))
        for (name, (i, j)) in eyebrows.items():
            for (x, y) in shape[i:j]:
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    cv2.imshow("Image", image)
    x = cv2.waitKey(10)
    userChar = chr(x & 0xFF)
    if userChar == 'q':
        break
#maybe try resizing the videos
#