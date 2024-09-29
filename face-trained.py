import os
import cv2 as cv 
import numpy as np
from tqdm import tqdm

people = ['Disha Patni', 'Elizabeth Olsen', 'Rashmika Mandanna']
DIR = 'Images'

features =[]
labels = []

def train_model():
    global people, DIR, features, labels
    try:
      haar_cascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_alt.xml')
    except Exception as e:
      print('please provide the Haarcascade model path',e)

    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in tqdm(os.listdir(path), colour='cyan'):
          if img.endswith('.jpg'):
            img_path = os.path.join(path, img)
            
            try:
              img_array = cv.imread(img_path)
              gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

              faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
              
              for (x, y, w, h) in faces_rect:
                  face_roi = gray[y:y+h, x:x+w]
                  features.append(face_roi)
                  labels.append(label)
                  
            except Exception as e:
              print("There is something wog with OpenCV",e)
              
              
opt = input('Do you want to train the model?')
if opt == 'y':
    train_model()
    print(f'length of features { len(features)}')
    print(f'length of labels { len(labels)}')

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features,labels)
    face_recognizer.save(f'face_trained{ len(labels)}.yml')

    np.save('features.npy', features)
    np.save('labels.npy', labels)

else:
  pass
