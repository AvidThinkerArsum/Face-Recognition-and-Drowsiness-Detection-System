import cv2
import face_recognition
from assist import SimpleFacerec

"""

### Using Existing Facial Landmark Indicators to find people from Images

img1 = cv2.imread("Images/Arsum0.jpg")
img2 = cv2.imread("Images/Keto0.jpg")

# Encoding image from bgr to rgb

rgb_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img_encoding1 = face_recognition.face_encodings(rgb_image1)[0]

rgb_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_image2)[0]

result = face_recognition.compare_faces([img_encoding1], img_encoding2)
print("Result: ", result)

cv2.imshow("IMG1", img1)
cv2.imshow("IMG2", img2)
cv2.waitKey(0)

### End of this Excercise

"""

"""

### Implementing the above in real time

# Encode all the Images:

help = SimpleFacerec()
help.load_encoding_images("Images/")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    
    # Detect Faces
    
    face_locations, face_names = help.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        
    cv2.imshow("IMG", frame)
    key = cv2.waitKey(1) # go to new frame every milisecond - live video
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

Result: I was oscialting between Arsum4, Arsum3, Arsum1. Would work well if have only one picture per person.
    
"""