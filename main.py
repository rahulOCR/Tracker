import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

param = cv.TrackerVit.Params()  
param.net= "../Vit/object_tracking_vittrack_2023sep.onnx" #model path
param.backend = cv.dnn.DNN_BACKEND_OPENCV                 # use CV backend
param.target =  cv.dnn.DNN_TARGET_CPU                     # use CPU

tracker = cv.TrackerVit.create(param)

roi = None

def visual(out, bbox, score, fps=None ):
    x, y, w, h = bbox


    if fps:
        cv.putText(out, 'FPS: {:.2f}'.format(fps), (0, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)

    cv.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 2)
    cv.putText(out, '{:.2f}'.format(score), (x, y+25), cv.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)
    return out

tic = cv.TickMeter()

while True:

    stat, frame = cap.read()

    if  not stat:
        print('Break')
        break
    frame = cv.resize(frame, (480,640))
    if roi is not None:
        tic.start()
        suc, bbox = tracker.update(frame)           #update
        score = tracker.getTrackingScore()
        tic.stop()
        if suc:
           frame= visual(frame,bbox,score,tic.getFPS())
        else:
            cv.putText(frame,'Target Lost!',(10,50), cv.FRONT_HERSHEY_DUPLEX,1,(0,0,255),1)

    cv.imshow('F1',frame)   
    key=cv.waitKey(1)
    
    if key == ord('s'):
        roi = cv.selectROI("F1",frame)
        tracker.init(frame,roi)         # Initialized
    
    if key == 27:
        break
    
cap.release()
cv.destroyAllWindows()