import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))

cap = cv2.VideoCapture(dir + '/Robots.mp4')

while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if not ret: break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    feat1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)
    feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat1, None)
    for i in range(len(feat1)):
        if np.linalg.norm(feat2[i]-feat1[i]) < 1:
            continue
        f10=int(feat1[i][0][0])
        f11=int(feat1[i][0][1])
        f20=int(feat2[i][0][0])
        f21=int(feat2[i][0][1])
        cv2.line(frame2, (f10,f11), (f20, f21), (0, 255, 0), 2)
        cv2.circle(frame2, (f10, f11), 5, (0, 255, 0), -1)

    cv2.imshow('robots', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
cap.release()
cv2.destroyAllWindows()
