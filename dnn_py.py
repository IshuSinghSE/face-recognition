# 
import cv2
import time

# video_capture = cv2.VideoCapture(0)
# time.sleep(2)

# -----------------------------------------------
# Face Detection using DNN Net
# -----------------------------------------------
# detect faces using a DNN model 
# download model and prototxt from https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models

def detectFaceOpenCVDnn(net, frame, conf_threshold=0.7):
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False,)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8,)
            # cv2.circle(frame, ((y2-x1)//2,(x2-y1)//2),(y2-y1), (0, 255, 0), int(round(frameHeight / 150)), 8,)
            # print(x1, x2, y1, y2)
            
            top=x1
            right=y1
            bottom=x2-x1
            left=y2-y1

            #  blurry rectangle to the detected face
            face = frame[right:right+left, top:top+bottom]
            face = cv2.GaussianBlur(face,(31, 31), 100)
            frame[right:right+face.shape[0], top:top+face.shape[1]] = face
            cv2.imshow('@ElBruno - Face Blur usuing DNN', frame)
            cv2.waitKey(0)  # waits until a key is pressed  
            cv2.destroyAllWindows()
    return frame, bboxes

# load face detection model
modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def start(image):
    try:
        # _, frameOrig = video_capture.read()
        img = cv2.imread(image)
        frame = cv2.resize(img, (640, 480))

        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net, frame)


    except Exception as e:
        print(f'exc: {e}')
        pass

    # key controller
    # key = cv2.waitKey(1) & 0xFF    
    # if key == ord("d"):
    #     detectionEnabled = not detectionEnabled

    # if key == ord("q"):
    #     break

# cv2.imshow('face detected',resized)
  