import cv2
import numpy as np
import time
import os
import base64
from telebot import *

net = cv2.dnn.readNet('C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/yolov4-tiny-custom.cfg','C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/yolov4-tiny-custom_best.weights')

classes = []
with open("C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/obj.names", "r") as f:
    classes = f.read().splitlines()
i=0

h , w =  None, None

writer = None
cap = cv2.VideoCapture(0)
#img = cv2.imread('brownspot.jpg')
api = '5484531444:AAHjI5X1W4zJ4QQm_3ruG-QPk0X6Kk-dH44'
bot = telebot.TeleBot(api)
receive_id = 1416579323

while True:
    
    _, img = cap.read()
    start_time=time.time()
    height, width ,_ =img.shape

    # Getting dimensions of the frame for once as everytime dimensions will be same
    if w is None or h is None:
        # Slicing and get height, width of the image
        h, w = img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                box_current = detection[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                
                boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                #boxes.append([x, y, w, h])
                #confidences.append((float(confidence)))
                #class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
     
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes)>0:
        for i in indexes.flatten():
            #x, y, w, h = boxes[i]
            x_min, y_min = boxes[i][0], boxes[i][1]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, [x_min, y_min, int(box_width), int(box_height)], color, 2)
            cv2.putText(img, label + " " + confidence, (x_min, y_min+20), font, 2, (255,255,255), 2)

    curent_time=time.time()
    inference_time=curent_time-start_time
    fps=1/inference_time
    print("inference time:" , inference_time)
    fps = int(fps)
    fps = str(fps)
    print("fps:" , fps)
    cv2.putText(img, fps, (7, 70), font, 3, (200, 255, 0), 3, cv2.LINE_AA) 
    cv2.imshow('Image', img)          
    key = cv2.waitKey(1)
    # Initialize writer
    if writer is None:
        resultVideo = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        writer = cv2.VideoWriter('result-video.mp4', resultVideo, 30,
                                 (img.shape[1], img.shape[0]), True)

    # Write processed current frame to the file
    writer.write(img)
    if key == 27:
        break 
    if key == ord('i'):
        cv2.imwrite(f'detect_{i}.jpg',img)
        i+=1
    if len(indexes)>0:
        detected = f'Kedetect_{i}.jpg'
        cv2.imwrite(detected,img)
        i+=1
        #chatid = message.chat.id
        bot.send_photo(receive_id, open(detected, 'rb'))

writer.release()
bot.polling()
cap.release()
cv2.destroyAllWindows