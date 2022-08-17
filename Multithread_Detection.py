from email.mime import base
import cv2
import numpy as np
import os
from time import sleep
import base64
import threading
from cryptography.fernet import Fernet
from telebot import *
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

#set parameter untuk source
cap = cv2.VideoCapture(0)
#img = cv2.imread('brownspot.jpg')


#manggil hasil training data
net = cv2.dnn.readNet('C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/yolov4-tiny-custom_best.weights', 'C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/yolov4-tiny-custom.cfg')

#membaca kelas by txt
classes = []
with open("C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/obj.names", "r") as f:
    classes = f.read().splitlines()

#declare untuk save video agar bisa append
i=0

#declare height, width 
h , w =  None, None

#set untuk save video hasil live
writer = None


#telegram key 
api = '5484531444:AAHjI5X1W4zJ4QQm_3ruG-QPk0X6Kk-dH44'
bot = telebot.TeleBot(api)
receive_id = 1416579323

layerOutputs = []

start_time=time.time()

_, frame = cap.read()

def trainingBackground():

    while True:

        #check jika source nya bernilai true 
        global _, frame
        _, frame = cap.read()

        #jika input adalah gambar
        height, width ,_ =frame.shape
        
        global w, h  

        # Getting dimensions of the frame for once as everytime dimensions will be same      
        if w is None or h is None:

            # Slicing and get height, width of the image
            h, w = frame.shape[:2]
        
        #set agar input size sesuai dengan hasil training
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

        net.setInput(blob)
        global start_time
        start_time=time.time()
        output_layers_names = net.getUnconnectedOutLayersNames()
        global layerOutputs
        layerOutputs = net.forward(output_layers_names)


def displayVideo():
    


    while True:
        #encoded byte64
        #global _, img
        #_, img = cap.read()
        #img = cv2.imencode('.jpg', img)[1].tobytes()
        #img = base64.encodebytes(img)
        #img = Fernet(key).encrypt(img)
        #img = frame.decode("utf-8")
        
        #mulai membuat bounding box
        boxes = []
        confidences = []
        class_ids = []

        if layerOutputs:
            #looping setiap layer output
            for output in layerOutputs:
                #looping setiap objek deteksi
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        box_current = detection[0:4] * np.array([w, h, w, h])

                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        #update setiap list dari bounding box kordinat, confidences, dan class_id
                        boxes.append([x_min, y_min, int(box_width), int(box_height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        #boxes.append([x, y, w, h])
                        #confidences.append((float(confidence)))
                        #class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2)
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))

            if len(indexes)>0:
                for i in indexes.flatten():
                    #x, y, w, h = boxes[i]
                    x_min, y_min = boxes[i][0], boxes[i][1]
                    label = str(classes[class_ids[i]])
                    print (label)
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(frame, (x_min, y_min), (x_min + int(box_width), y_min + int(box_height)), color, 2)
                    cv2.putText(frame, label + " " + confidence, (x_min, y_min-5), font, 2, (255,255,255), 2)
                
            #key = b'sPy0VTSyePWQTR7mDmOeJbk6JWS5LfGyO0OJ7uiJxE8=' 
            #key = Fernet.generate_key()

            #menampilkan inference dan fps time 
            curent_time=time.time()
            if curent_time > start_time:              
                inference_time=curent_time-start_time
                fps=1/inference_time
                print("inference time:" , inference_time)
                fps = int(fps)
                fps = str(fps)
                print("fps:" , fps)
                #cv2.putText(frame, fps, (7, 70), font, 3, (200, 255, 0), 3, cv2.LINE_AA) 
                
                #cv2.imshow('Image', frame) 
                ##ke byte
                #f = Fernet(key)
                #frames_encoded = cv2.imencode('.jpg', frame)[1].tobytes()
                #token = f.encrypt(frames_encoded)
                #data = Fernet(key).encrypt(frames_encoded)
                #data = base64.encodebytes(data)
                #data = Fernet(key).decrypt(data)
                #frames = data.decode("utf-8")
                #print(data)
                
                cv2.imshow('Image', frame)
                
                password = b"test"
                salt = os.urandom(16)                
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(),
                 length=32,
                 salt=salt,
                 iterations=100000,
                 )

                key = base64.urlsafe_b64encode(kdf.derive(password)) 
                f = Fernet(key)
                frames_encoded = cv2.imencode('.jpg', frame)[1].tobytes()
                token = f.encrypt(frames_encoded)
                #token = f.decrypt(token)
                #f.decrypt(token)
                frames = np.frombuffer(token, np.byte)
                print(key)
                #cv2.imshow('Image', frames) 
                key_2 = cv2.waitKey(1)
                
                #key input untuk exit dan save gambar 
                if key_2 == 27:
                    return False 
                if key_2 == ord('i'):
                    cv2.imwrite(f'detect_{i}.jpg',frame)
                    i = 0
                    i+=1
                if len(indexes)>0:
                    detected = f'Kedetect_{i}.jpg'
                    cv2.imwrite(detected,frame)
                    i = 0
                    i+=1
                    ##chatid = message.chat.id
                    bot.send_photo(receive_id, open(detected, 'rb'))
                    bot.send_message(receive_id,'Status padi ini: ' + label)
                    
    bot.polling()
    cap.release()
    cv2.destroyAllWindows
    



if __name__ == '__main__':
    c = threading.Thread(name='trainingBackground', target=trainingBackground)
    c.daemon = True
    c.start()
    
    #t1 = threading.Thread(name='trainingBackground', target=trainingBackground)
    #t2 = threading.Thread(name='displayVideo', target=displayVideo)
    
    #t1.start()
    #t2.start()
    
    #t2.join()
                          
    displayVideo()