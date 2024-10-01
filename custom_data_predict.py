import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'su_alti_circle.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

#cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture('input_vid.mp4')
#cap = cv2.VideoCapture(0)
ret, frame = cap.read()
H, W, _= frame.shape
#out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MJPG'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
out = cv2.VideoWriter('output_vid.AVI', cv2.VideoWriter_fourcc(*'MJPG'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model_path = os.path.join('.', 'runs', 'detect', 'yolov8n_custom', 'weights', 'best.pt') 

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5


#results = model("SUNP0253.jpg")
#print(results)

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        x_center = int((x1 + x2) / 2) #x cordinate of center
        y_center = int((y1 + y2) / 2) #y cordinate of center 

        if score > threshold: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(frame, (x_center,y_center), 10,(0, 255, 0) , -1)

        '''if class_id == 0:# 0 indice classe here is a latuce plante (marul bitki)
            x_center = int((x1 + x2) / 2) #x cordinate of center
            y_center = int((y1 + y2) / 2) #y cordinate of center 
            
            if score > threshold: 
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(frame, (x_center,y_center), 10,(0, 255, 0) , -1)

        if class_id == 1:# 0 indice classe here is a bad plante (yabanci bitki)
            x_center = int((x1 + x2) / 2) #x cordinate of center
            y_center = int((y1 + y2) / 2) #y cordinate of center 
            
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, (x_center,y_center), 10,(0, 0, 255) , -1)'''
    out.write(frame)
    cv2.imshow('frame', frame) 
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
