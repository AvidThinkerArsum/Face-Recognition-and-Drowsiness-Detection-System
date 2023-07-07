# Import Libraries
# Explaining Libraries:

# cv2 for impage processing
# dlib 68 points face landmark detector using ML
# imutils for image processing
# scipy and numpy have some functionality
# scipy written in Python has vaster functionality
# numpy written in C is faster computationally

import cv2
import sys
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
from pygame import mixer

# takes eye as 6 point array and finding EAR. An EAR of 0 means eyes are closed.

def Eye_Aspect_Ratio (eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A+B)/(2*C)
    return ear

mixer.init()
mixer.music.load("alarmcopy.wav")

threshold = 0.20
flag = 0
num_frames = 20
    
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye'] # taking left eye from the documentation
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye'] # taking right eye fromt the documentation

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, height = 800, width = 900)
    
    if ret == True:
        pass
    else:
        print("Unable to read frame.")
        sys.exit()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray) # detecting face
    
    for face in faces:
        
        landmark = predictor(gray, face) # finding all features on face
        landmark = face_utils.shape_to_np(landmark) # converting detected features to array
        
        leftEye = landmark[lStart:lEnd] # extracting the features for the left eye from array
        rightEye = landmark[rStart:rEnd] # extracting the features for the right eye from array
        
        leftEar = Eye_Aspect_Ratio(leftEye) # a six entry array for left eye
        righttEar = Eye_Aspect_Ratio(rightEye) # a six entry array for right eye
        
        leftEyeHull = cv2.convexHull(leftEye) # drawing set of six points over the left eye
        rightEyeHull = cv2.convexHull(rightEye) # drawing set of six points over the right eye
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) # drawing outline around left eye
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) # drawing outline around right eye
        #  (frame, the contour itself, which points you want - we want all, color, thickness)
        
        if (leftEar < threshold) and (righttEar < threshold): # eyes closed
            flag += 1
            print (flag)
            if flag >= num_frames:
                cv2.putText(frame, "***** ALERT *****", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # (frame, message, position, font type, font scale, line color, font thickness)
                mixer.music.play()
        else: # if not closed continuously
            flag = 0
            
    cv2.imshow("Drowsiness Detection", frame) # to draw the frame continuously
    
    if cv2.waitKey(1) == 27: # Escape Key
        break 
    
cv2.destroyAllWindows()
cap.release()

        
        