import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os
from pyzbar.pyzbar import decode

# Step 1: Read Images
path = 'ImageData'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    currImg = cv2.imread(f'{path}/{cls}')  # Read each image in the Folder
    images.append(currImg)  # Append each each to the list
    classNames.append(os.path.splitext(cls)[0])

print(classNames)


# Step 2: Encoding process
def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        namesList = []
        for line in myDataList:
            entry = line.split(',')
            namesList.append(entry[0])
        if (name not in namesList):
            nowTime = datetime.now()
            dtString = nowTime.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# Barcode Reader
def read_barcodes(frame):
    barcodes = decode(frame)
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        # 1
        barcode_info = barcode.data.decode('utf-8')
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 2
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, barcode_info, (x + 6, y), font, 2.0, (0, 255, 0), 1)
        print(barcode_info)

    return frame


encodeListKnown = find_encodings(images)
print('Encoding Complete')

# Step 3: Find Matches with images coming from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # Capture frame by frame
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize to small image by mentioning the scale
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # Convert to RGB

    '''Counting Number of Faces'''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_small)
    numFaces = len(faces)

    img = read_barcodes(img)

    facesCurrentFrame = face_recognition.face_locations(img_small)  # Get all the face locations in the image
    encodesCurrFrame = face_recognition.face_encodings(img_small, facesCurrentFrame)  # Encodings of all the faces

    # Finding the matches of encodings of faces in the current frame to all the encodings we have found before
    for encodeFace, faceLocation in zip(encodesCurrFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # print(matches)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:  # For the match found
            name = classNames[matchIndex].upper()  # Extract the name of the match
            # print(name)
            y1, x2, y2, x1 = faceLocation  # Extract the face box location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)
        else:
            y1, x2, y2, x1 = faceLocation  # Extract the face box location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'NOT IN DATABASE', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
# 3
cap.release()
cv2.destroyAllWindows()
