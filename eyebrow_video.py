import numpy as np
import argparse
import imutils
import dlib
import cv2
import copy
pp = "shape_predictor_68_face_landmarks.dat"

facial_points = {
    "mouth": (48, 68), "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "nose": (27, 35),
    "jaw": (0, 17),
}
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

def within_range(difference_threshold,a,b):
    differences =[]
    for i in range(4):
        differences.append(abs(a[i]-b[i]))
    if(differences[0]<difference_threshold and differences[1]<difference_threshold and differences[2]<difference_threshold and differences[3]<difference_threshold):
        return True
    return False


def percent_checker(x,y):
    return (x/y)

def distance(x1,y1,x2,y2):
    y = y2-y1
    x = x2-x1
    return pow((pow(y,2) + pow(x,2)), 0.5)

def rect_to_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()
    h = rect.bottom()
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pp)
distances = {
"left_eyebrow1": 10,
"left_eyebrow2": 10,
"right_eyebrow1": 10,
"right_eyebrow2": 10
}

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
final_faces = []
for i in range(10):
    ret1, image1 = cap.read(0)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Example", gray)
    # cv2.waitKey()
    faces = detector(gray, 1) #facial region
    if(len(faces)>len(final_faces)):
        final_faces = faces
cap.release()
print(final_faces)
print(final_faces[0])
temp_faces = final_faces
final_faces= []
for i in range(len(temp_faces)):
    replacement = [rect_to_box(temp_faces[i]),0]
    final_faces.append(replacement)
cap = cv2.VideoCapture(0)
print(final_faces)
while True:
    ret, image = cap.read()
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Example", gray)
    # cv2.waitKey()
    rects = detector(gray, 1) #facial region
    final_faces_copy = copy.deepcopy(final_faces)
    for (i, rect) in enumerate(rects): #looping through the information
        shape = predictor(gray, rect)#facial points
        shape = shape_to_np(shape) #array that produces x,y values for each numbered point
        (x, y, w, h) = rect_to_box(rect) #draws bounding box
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
        face_number = 100;
        for i in range (len(final_faces_copy)):
            if (within_range(50,(x,y,w,h),final_faces_copy[i][1])):
                face_number = i
                break
        temp_eyedistance ={}
        for i in range(1,6):
            distances2["eyebrow"+str(i)] = distance(shape[2+i][0], shape[2+i][1], shape[16+i][0], shape[16+i][1])
            cv2.line(image,(shape[2+i][0],shape[2+i][1]),(shape[16+i][0], shape[16+i][1]),(255,0,0),2)
            distances2["eyebrow"+str(11-i)] = distance(shape[14-i][0], shape[14-i][1], shape[27-i][0], shape[27-i][1])
            cv2.line(image,(shape[14-i][0], shape[14-i][1]),(shape[27-i][0], shape[27-i][1]),(255,0,0),2)
        for i in range(1, 3):
            temp_eyedistance["lefteye" + str(i)] = distance(shape[36 + i][0], shape[36 + i][1], shape[42 - i][0],
                                                      shape[42 - i][1])
            temp_eyedistance["righteye" + str(i)] = distance(shape[42 + i][0], shape[42+i][1], shape[48 - i][0],
                                                           shape[48 - i][1])


            eyesdistance["lefteye" + str(i)] = distance(shape[36 + i][0], shape[36 + i][1], shape[42 - i][0],
                                                      shape[42 - i][1])

            cv2.line(image, (shape[36 + i][0], shape[36 + i][1]), (shape[42 - i][0],
                                                      shape[42 - i][1]), (0, 255, 0), 2)

            eyesdistance["righteye" + str(i)] = distance(shape[42 + i][0], shape[42+i][1], shape[48 - i][0],
                                                           shape[48 - i][1])
            cv2.line(image, (shape[42 + i][0], shape[42+i][1]), (shape[48 - i][0],
                                                           shape[48 - i][1]), (0, 255, 0), 2)


        for (name, (i, j)) in eyebrows.items():
            for (x, y) in shape[i:j]:
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    cv2.imshow("Image", image)
    x = cv2.waitKey(10)
    userChar = chr(x & 0xFF)
    if userChar == 'q':
        break
# output = visualize_facial_landmarks(image, shape)
# cv2.imshow("Image Final", output)
# cv2.waitKey(0)
