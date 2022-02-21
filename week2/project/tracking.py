import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans

dir = os.path.dirname(os.path.realpath(__file__))

cap = cv2.VideoCapture(dir + '/Challenge.mp4')
ret, frame1 = cap.read()

# crop_img = frame1[500:850, 1500:1750]
crop_img = frame1[530:850,975:1150]
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(crop_img,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(crop_img, kp1)

bf = cv2.BFMatcher()

def distance(pt1,pt2):
    return np.linalg.norm([pt2[0]-pt1[0],pt2[1]-pt1[1]])

l = 0
while cap.isOpened():
    l += 1
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if not ret: break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    feat1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)
    feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat1, None)
    moving = []
    for i in range(len(feat1)):
        if np.linalg.norm(feat2[i]-feat1[i]) < 1:
            continue
        f20=int(feat2[i][0][0])
        f21=int(feat2[i][0][1])
        moving.append([f20,f21])

    moving = np.array(moving)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(moving)
    
    crop1 = frame2[
        max(int(kmeans.cluster_centers_[0][1])-150,0):min(int(kmeans.cluster_centers_[0][1])+150,frame2.shape[0]),
        max(int(kmeans.cluster_centers_[0][0])-150,0):min(int(kmeans.cluster_centers_[0][0])+150,frame2.shape[1])]
    kpcp1 = orb.detect(crop1,None)
    kpcp1, descp1 = orb.compute(crop1, kpcp1)

    crop2 = frame2[
        max(int(kmeans.cluster_centers_[1][1])-150,0):min(int(kmeans.cluster_centers_[1][1])+150,frame2.shape[0]),
        max(int(kmeans.cluster_centers_[1][0])-150,0):min(int(kmeans.cluster_centers_[1][0])+150,frame2.shape[1])]
    kpcp2 = orb.detect(crop2,None)
    kpcp2, descp2 = orb.compute(crop2, kpcp2)

    MIN_DIST = 250
    matches = bf.match(des1, descp1)
    matches = sorted(matches, key = lambda x:x.distance)
    # if len(matches) > 0:
    #     img3 = cv2.drawMatches(crop_img,kp1,crop1,kpcp1,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     plt.figure(1, figsize = (20,20))
    #     plt.imshow(img3)
    #     plt.savefig(f'test/crop1_{l}.png')
    crop1_good_matches = 0
    for m in matches:
        if m.distance < MIN_DIST: 
            crop1_good_matches += 1
    matches = bf.match(des1, descp2)
    matches = sorted(matches, key = lambda x:x.distance)
    # if len(matches) > 0:
    #     matches = sorted(matches, key = lambda x:x.distance)
    #     img3 = cv2.drawMatches(crop_img,kp1,crop2,kpcp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     plt.figure(2, figsize = (20,20))
    #     plt.imshow(img3)
    #     plt.savefig(f'test/crop2_{l}.png')
    crop2_good_matches = 0
    for m in matches:
        if m.distance < MIN_DIST: 
            crop2_good_matches += 1

    MIN_GOOD_MATCH = 5
    if crop1_good_matches > crop2_good_matches and crop1_good_matches > MIN_GOOD_MATCH:
        des1 = descp1
        cv2.circle(frame2, (int(kmeans.cluster_centers_[0][0]),int(kmeans.cluster_centers_[0][1])), 50, (147,20,255), -1)
    elif crop2_good_matches > MIN_GOOD_MATCH: 
        des1 = descp2
        cv2.circle(frame2, (int(kmeans.cluster_centers_[1][0]),int(kmeans.cluster_centers_[1][1])), 50, (147,20,255), -1)

    cv2.imshow('crop1', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
