import cv2 as cv
import numpy as np
import shutil, os
import time
from tqdm import tqdm

IMAGE = 'Images/'
RESULT = 'Group'

face_cascade=cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
count = 0
for folder in os.listdir(IMAGE):
    
    RESULTS = os.path.join(RESULT,folder)    
    
    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)
        print(RESULTS)
            
    for images in tqdm(os.listdir(os.path.join(IMAGE,folder))): 
      if images.endswith('.jpg'):
        images = os.path.join(IMAGE,folder,images)
        img = cv.imread(images)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(img, 1.1, 4)
                        
        for (x, y, w, h) in faces: 
            # Draw rectangle around the faces
            cv.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
            
        if len(faces) > 2:   
            shutil.move(images,RESULTS)
            # print(len(faces))
            # cv.imshow('Face Recognized', img)
            # time.sleep(1)
            # cv.waitKey(0)
            count+=1
        #   # waits until a key is pressed  
        # cv.destroyAllWindows()  # destroys the window showing image  
    print(count, 'images sorted!')