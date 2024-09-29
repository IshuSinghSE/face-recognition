import numpy as np
import cv2 as cv 
from tqdm import tqdm
import os

def Resize(path, scale=1, width=0, height=0, show=False, color=1):
    IMAGE = cv.imread(path, color)
    width = int(IMAGE.shape[1]*scale)
    height = int(IMAGE.shape[0]*scale)
    dimension = (width, height)
    resized = cv.resize(IMAGE, dimension, interpolation=cv.INTER_AREA)
    if show:
        cv.imshow(f'Resized {resized.shape}',resized)
        
    return resized


def recognize(image):
    global face_recognizer, people, haar_cascade
    
    # features = np.load('features.npy', allow_pickle=True)
    # labels = np.load('labels.npy', allow_pickle=True)
    scale = 0.4
    img = cv.imread(image)
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dimension = (500, 500)
    resized=cv.resize(img, dimension, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    # cv.imshow('person', gray)
    # cv.waitKey(0)
    face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x,y,w,h) in face_rect:
        faces_roi = gray[y:y+h, x:x+h]
        label, confidence = face_recognizer.predict(faces_roi)
        print(f' * {people[label]} - { confidence}')

        cv.putText(resized, str(people[label]), (x, y-10), cv.FONT_HERSHEY_COMPLEX,0.7, (0,0,255), thickness=1)
        cv.rectangle(resized, (x,y), (x+w,y+h), (100, 0, 255),  thickness=2)

    cv.imshow('Face Recognized', resized)
    cv.waitKey(0)  # waits until a key is pressed  
    cv.destroyAllWindows()  # destroys the window showing image  

path = 'Images-testing'
train_folder = 'Images'
people = os.listdir(train_folder)#['Elizabeth Olsen', 'Rashmika Mandanna','unknown']

print('Loading model...')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained3884.yml')
haar_cascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')

for folder in os.listdir(path):
  if folder == 'Elizabeth Olsen': 
    for image in os.listdir(os.path.join(path,folder)):
      if image.endswith('.jpg'):
        recognize(os.path.join(path,folder,image))
    print(folder)
    