import cv2
import numpy as np
import face_recognition

# Step 1 - Loading images and covert to RGB
img_elon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
img_elon = cv2.cvtColor(img_elon,cv2.COLOR_BGR2RGB)

img_elon_test = face_recognition.load_image_file('ImagesBasic/Elon Test.jpg')
img_elon_test = cv2.cvtColor(img_elon_test,cv2.COLOR_BGR2RGB)


# Step 2 - Find faces in the image and their encodings

faceLoc = face_recognition.face_locations(img_elon)[0]
# Find the location of the face - 4D coordinates
encodeElon = face_recognition.face_encodings(img_elon)[0]
# print(encodeElon)
cv2.rectangle(img_elon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# Draw a rectangle around the face by specifying the location points and specify the colour and thickness


faceLocTest = face_recognition.face_locations(img_elon_test)[0]
# Find the location of the face - 4D coordinates
encodeElonTest = face_recognition.face_encodings(img_elon_test)[0]
# print(encodeElon)
cv2.rectangle(img_elon_test,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
# Draw a rectangle around the face by specifying the location points and specify the colour and thickness

# Step 3 - Comparing the faces and finding the distance between them using their encodings

results = face_recognition.compare_faces([encodeElon],encodeElonTest)
print(results)
# Result is True means the encodings are of the same person, if we replace test image with Bill Gates then result is False

# Find the best match by finding the distance ~ lower the distance better the match
faceDist = face_recognition.face_distance([encodeElon],encodeElonTest)
print(faceDist)

cv2.putText(img_elon_test, f'{results} {np.round(faceDist,2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Elon Musk', img_elon)
cv2.imshow('Elon Musk Test', img_elon_test)
cv2.waitKey(0)

