import cv2
import numpy as np
import face_recognition
import os
import requests
from datetime import datetime

# reading database images
path='Images'
images=[]
classNames=[]
myList= os.listdir(path)

for i in myList:
    curImg=cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])

# now finding encodings
def findEncocding(images):
    List=[]
    for i in images:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(i)[0]
        List.append(encode)
    return List

# function to mark attendance
def markAttendance(name):
    with open('attendance.csv','r+')as f:
        myDataList =f.readlines()
        nameList =[]
        for line in myDataList:
            entry =line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now =datetime.now()
            dtstring =now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

# func calling
encodeList = findEncocding(images)

#VideoCam
cap=cv2.VideoCapture(0)

# using ipcam in android phone
url = 'http://192.168.43.173:8080/shot.jpg'


while True:
    # for laptop cam
    # success,img=cap.read()
    # !=============!

    # for android phone
    img_resp = requests.get(url)
    img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
    img=cv2.imdecode(img_arr,-1)

    imgresize =cv2.resize(img,(0,0),None,0.25,0.25)
    imgresize =cv2.cvtColor(imgresize,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgresize)
    encodeCurFrame=face_recognition.face_encodings(imgresize,facesCurFrame)

    for i,j in zip(encodeCurFrame,facesCurFrame):
        match = face_recognition.compare_faces(encodeList,i)
        facedis =face_recognition.face_distance(encodeList,i)
        matchidx=np.argmin(facedis)

        if match[matchidx]:
            name=classNames[matchidx].upper()
            print(name)
            markAttendance(name)

    cv2.imshow("Cam",imgresize)
    if cv2.waitKey(1)==ord('q'):
        break

print("Attendance Marked")



