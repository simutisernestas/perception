import numpy as np
import cv2 as cv2
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import time as t
from helpers import *



def extract_keypoints_surf(img1, img2, K, baseline):
    """
    use sift to detect keypoint features
    remember to include a Lowes ratio test
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img1,None)
    match_points1 = [x.pt for x in kp]
    kp, des = sift.detectAndCompute(img2,None)
    match_points2 = [x.pt for x in kp]

    p1 = np.array(match_points1).astype(float)
    p2 = np.array(match_points2).astype(float)

    ##### ############# ##########
    ##### Do Triangulation #######
    ##### ########################
    #project the feature points to 3D with triangulation
    
    #projection matrix for Left and Right Image
    M_left = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    M_rght = K.dot(np.hstack((np.eye(3), np.array([[-baseline, 0, 0]]).T)))

    p1_flip = np.vstack((p1.T, np.ones((1, p1.shape[0]))))
    p2_flip = np.vstack((p2.T, np.ones((1, p2.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])

    # Normalize homogeneous coordinates (P->Nx4  [N,4] is the normalizer/scale)
    P = P / P[3]
    land_points = P[:3]

    return land_points.T, p1

def featureTracking(img_1, img_2, prev_points, world_points):
    params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    next_points, status, error = cv2.calcOpticalFlowPyrLK(img_1, img_2, prev_points.astype(np.float32), None, **params)

    status = status.reshape(status.shape[0])

    world_points = world_points[status == 1]
    prev_points = prev_points[status == 1]
    next_points = next_points[status == 1]

    return world_points, prev_points, next_points

def playImageSequence(left_img, right_img, K):

    baseline = 0.54

    ##### ################################# #######
    ##### Get 3D points Using Triangulation #######
    ##### #########################################
    """
    Implement step 1.2 and 1.3
    Store the features in 'reference_2D' and the 3D points (landmarks) in 'landmark_3D'
    hint: use 'extract_keypoints_surf' above
    """

    # reference
    reference_img = left_img

    # Groundtruth for plot
    truePose = getTruePose()
    traj = np.zeros((600, 600, 3), dtype=np.uint8);
    maxError = 0

    for i in range(0, 1400):
        print('image: ', i)
        curImage = getLeftImage(i)
        curImage_R = getRightImage(i)

        ##### ############################################################# #######
        ##### Calculate 2D and 3D feature correspndances in t=T-1, and t=T  #######
        ##### #####################################################################
        """
        Implement step 2.2)
        Remember this is a part of a loop, so the initial features are already
        provided in step 1)-1.3) outside the loop in 'reference_2D' and 'landmark_3D'
        """
        land_points, p1 = extract_keypoints_surf(curImage, curImage_R, K, baseline)
        world_points, prev_points, next_points = featureTracking(curImage, curImage_R, p1, land_points)

        ##### ################################# #######
        ##### Calculate relative pose using PNP #######
        ##### #########################################
        """
        Implement step 2.3)
        """
        _ , rvec, tvec, inliers = cv2.solvePnPRansac(landmark_3D, next_points, K, None)

        ##### ####################################################### #######
        ##### Get Pose and Tranformation Matrix in world coordionates #######
        ##### ###############################################################
        rot, _ = cv2.Rodrigues(rvec)
        tvec = -rot.T.dot(tvec)  # coordinate transformation, from camera to world. What is the XYZ of the camera wrt World
        inv_transform = np.hstack((rot.T, tvec))  # inverse transform. A tranform projecting points from the camera frame to the world frame

        ##### ################################# #######
        ##### Get 3D points Using Triangulation #######
        ##### #########################################
        # re-obtain the 3D points
        """
        Implement step 2.4)
        """
        # TODO:
        
        #Project the points from camera to world coordinates
        reference_2D = reference_2D_new.astype('float32')
        landmark_3D = inv_transform.dot(np.vstack((landmark_3D_new.T, np.ones((1, landmark_3D_new.shape[0])))))
        landmark_3D = landmark_3D.T

        ##### ####################### #######
        ##### Done, Next image please #######
        ##### ###############################
        reference_img = curImage

        ##### ################################## #######
        ##### START OF Print and visualize stuff #######
        ##### ##########################################
        # draw images
        draw_x, draw_y = int(tvec[0]) + 300, 600-(int(tvec[2]) + 100);
        true_x, true_y = int(truePose[i][3]) + 300, 600-(int(truePose[i][11]) + 100)

        curError = np.sqrt(
            (tvec[0] - truePose[i][3]) ** 2 +
            (tvec[1] - truePose[i][7]) ** 2 +
            (tvec[2] - truePose[i][11]) ** 2)
        
        if (curError > maxError):
            maxError = curError

        print(tvec[0],tvec[1],tvec[2], rvec[0], rvec[1], rvec[2])
        print([truePose[i][3], truePose[i][7], truePose[i][11]])
        
        text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(tvec[0]), float(tvec[1]),
                                                                           float(tvec[2]));
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, 255), 2);
        cv2.circle(traj, (true_x, true_y), 1, (255, 0, 0), 2);
        cv2.rectangle(traj, (10, 30), (550, 50), (0, 0, 0), cv2.FILLED);
        cv2.putText(traj, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8);

        h1, w1 = traj.shape[:2]
        h2, w2 = curImage.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1, :3] = traj
        vis[:h2, w1:w1 + w2, :3] = np.dstack((np.dstack((curImage,curImage)),curImage))

        cv2.imshow("Trajectory", vis);
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Maximum Error: ', maxError)
    ##### ################################ #######
    ##### END OF Print and visualize stuff #######
    ##### ########################################

if __name__ == '__main__':
    left_img = getLeftImage(0)
    right_img = getRightImage(0)

    K = getK()

    playImageSequence(left_img, right_img, K)