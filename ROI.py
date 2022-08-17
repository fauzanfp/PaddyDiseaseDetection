import cv2 
import numpy as np 
import time 

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet('C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/yolov4-tiny-custom.cfg','C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/yolov4-tiny-custom_best.weights')

classes = []
with open("C:/Users/Acer/Documents/yolov4-tiny-custom_Training-main/yolov4-tiny/yolov4-tiny_new_weight/obj.names", "r") as f:
    classes = f.read().splitlines()

h , w =  None, None



while True:

    ret, frame = cap.read()
    
    if w is None or h is None:
        h, w = frame.shape[:2]
    
    
    x1 = 400
    y1 = 10

    x2 = 610 
    y2 = 250
    
    roi = frame[y1:y2, x1:x2]

    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 480), (0,0,0), swapRB=True, crop=False)

    net.setInput(blob)
    start_time=time.time()
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
                #x_min = int(x_center - (box_width / 2))
                #y_min = int(y_center - (box_height / 2))
                x_min = int(x1- (y1 / 2))
                y_min = int(x2 - (y2 / 2))
                
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
            print(label)
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(frame, [x_min, y_min, int(y1), int(y2)], color, 2)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), color, 1)
            cv2.putText(frame, label + " " + confidence, (x_min, y_min+20), font, 2, (255,255,255), 2)

    
    curent_time=time.time()
    inference_time=curent_time-start_time
    fps=1/inference_time
    print("inference time:" , inference_time)
    fps = int(fps)
    fps = str(fps)
    print("fps:" , fps)
    #cv2.imshow("frame", frame)
    cv2.imshow("roi", roi)
    key = cv2.waitKey(1)
    #interrupt = cv2.waitKey(0)
    #if interrupt & 0xFF == 27:
    #    break
    if key == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()