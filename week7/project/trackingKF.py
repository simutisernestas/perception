from tkinter import Frame
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
cap = cv2.VideoCapture('rolling_ball.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/fps
dt

def update(x, P, Z, H, R):
    y = Z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.pinv(S)
    Xprime = x + K @ y
    KH = K @ H
    Pprime = (np.eye(KH.shape[0]) - KH) @ P
    return (Xprime,Pprime)
    
def predict(x, P, F, u):
    Xprime = F @ x + u
    Pprime = F @ P @ F.T
    return (Xprime,Pprime)
    
### Initialize Kalman filter ###
# The initial state (6x1).
# x y x_dt y_dt x_dt2 y_dt2
x = np.array([[600,200,0,0,0,0]]).T

# # The initial uncertainty (6x6).
P = np.array([
    [100,0,0,0,0,0],
    [0,100,0,0,0,0],
    [0,0,100,0,0,0],
    [0,0,0,100,0,0],
    [0,0,0,0,100,0],
    [0,0,0,0,0,100],
])

# # The external motion (6x1).
u = np.array([[0,0,0,0,0,0]]).T

# The transition matrix (6x6). 
F = np.eye(6)
F[0][2] = dt
F[1][3] = dt
F[2][4] = dt**2
F[3][5] = dt**2

# # The observation matrix (2x6).
H = np.zeros((2,6))
H[0][0] = 1
H[1][1] = 1

# # The measurement uncertainty.
R = 1

path = []
# Looping through all the frames
counter = 0
while True:
    counter += 1
    ret, frame = cap.read()
    if counter < 15:
        continue
    if counter > 52:
        break
    print(f"frame: {counter}")
    if not ret: 
        break
    factor = 3
    frame = cv2.resize(frame, (int(frame.shape[1]/factor), int(frame.shape[0]/factor)))
    orgf = frame.copy()
    ### Detect the ball ###
    frame[(frame[:,:,0] < 5) | (frame[:,:,0] > 60)] = 0
    circles = cv2.HoughCircles(frame[:,:,0], cv2.HOUGH_GRADIENT_ALT, 1, 1, param1=300, param2=.75, minRadius=10, maxRadius=30)
    ### If the ball is found, update the Kalman filter ###
    if circles is not None:
        circles = np.uint16(np.around(circles))
        i = circles[0,0]
        # cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
        Z = np.array([[i[0],i[1]]]).T
        x,P = update(x, P, Z, H, R)
        cv2.circle(orgf,(int(x[0][0]),int(x[1][0])), 30, (0,0,255), -1)
        path.append(x)
    
    ### Predict the next state
    x,P = predict(x, P, F, u)

    ### Draw the current tracked state and the predicted state on the image frame ###
    cv2.circle(orgf,(int(x[0][0]),int(x[1][0])), 30, (255,0,0), -1)
    cv2.imshow('image',orgf)
    cv2.waitKey(30)

cv2.destroyAllWindows()