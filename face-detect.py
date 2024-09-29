# Face Detection using Open-CV
import cv2
import PIL.Image
import os
from tqdm import tqdm
from time import sleep

# print(os.getcwd())
# Load the cascade
XML = "haarcascades"
IMAGE = "Images"
RESULT = "Result/"
COUNT = 0

# selecting model for list of available models
for model in os.listdir(XML):
        model_path = os.path.join(XML,model)
        model = model.strip('.xml').replace('haarcascade_','').capitalize()
        print(f' * {model} model selected !')
        face_cascade=cv2.CascadeClassifier(model_path)
        
        #selecting folder of immage of specific person
        for dirs in os.listdir(IMAGE):
            print(f' ^ {dirs} person selected !')
            # selecting image of the selected person
            
            for image in os.listdir(os.path.join(IMAGE , dirs)):
                picture = os.path.join(os.path.join(IMAGE , dirs),image)
                # print(image, 'pictures selected !')
                
                # Read the input image
                img = cv2.imread(picture)
                # Detect faces
                faces = face_cascade.detectMultiScale(img, 1.1, 4)
                
                for (x, y, w, h) in faces: 
                    # Draw rectangle around the faces
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
                    
                    folder = os.path.join(RESULT,dirs, model)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        print(folder, 'created! ')
                        
                    # Display the output
                    cv2.imwrite(os.path.join(RESULT,dirs,model,image), img)
                    COUNT+= 1
                    
print(f'''{len(XML)} models used ! \n {len(os.listdir(IMAGE))} unique people detected  \n *** < {COUNT} > *** images processed !!!'''  )   
print("Everything ihas been done succesfully!!!")
            

# 
# Real=PIL.Image.open(r"Results/face_detected.jpg")