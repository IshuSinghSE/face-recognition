import cv2
import os
import numpy as np
from dnn_py import start
modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "models/deploy.prototxt"

def detect(image):
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    img = cv2.imread(image)
    
    resized = cv2.resize(img, (300, 300))
    h, w = resized.shape[:2]
    blob = cv2.dnn.blobFromImage(resized, 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw faces on image
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(resized, (x, y), (x1, y1), (100, 0, 255), 1) 
                cv2.imshow('face detected',resized)
                cv2.waitKey(1000)  # waits until a key is pressed  
                cv2.destroyAllWindows()  # destroys the window showing image  

path = 'Images-testing'

for folder in os.listdir(path):
  if folder == 'Rashmika Mandanna': 
    for image in os.listdir(os.path.join(path,folder)):
      if image.endswith('.jpg'):
        start(os.path.join(path,folder,image))
    print(folder)