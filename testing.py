#HaarCascade Built-in Classifiers

import cv2
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

vidCap = cv2.VideoCapture(0)
ret1,img1 = vidCap.read()
while True:
    ret, img = vidCap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #change figures later
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),  (255,0,0),2)
        roi_face = gray[y:y+h, x:x+w]
        roi_orig = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_face)
        for (ex,ey,ew,eh)in eyes:
            cv2.rectangle(roi_orig,(ex,ey),(ex+ew,ey+eh), (0,255,0), 2)
    cv2.imshow('face detection', img)
    x = cv2.waitKey(10)
    userChar = chr(x & 0xFF)
    if userChar == 'q':
        break
vidCap.release()
cv2.destroyAllWindows()
