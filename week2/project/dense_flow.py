import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))

cap = cv2.VideoCapture(dir + '/Robots.mp4')
ret, frame1 = cap.read()
pink = np.array([147,20,255])

while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if not ret: break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 1, 15, 3, 5, 1.5, 0)
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    mask = (mag > 7)
    frame2[mask] = pink

    cv2.imshow('robots', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
cap.release()
cv2.destroyAllWindows()
