import cv2
import face_recognition

### Using Existing Facial Landmark Indicators to find people from Images
***

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

### Implementing the above in real time