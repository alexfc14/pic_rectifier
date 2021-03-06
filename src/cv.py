import numpy as np
import cv2 as cv

import streamlit as st

# def calibrate(images, grid_w=9, grid_h=6, verbose=True):
def calibrate(images, grid_w=49, grid_h=24, verbose=True):
    # 1. DETECT KEYPOINTS
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(grid_h,5,0)
    objp = np.zeros((grid_h*grid_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_w,0:grid_h].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    calibration_shape = None
    for name, img in images.items():
        if verbose:
            st.text(name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if verbose:
            st.image(img, width=100)
        calibration_shape = calibration_shape or gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (grid_w,grid_h), cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_ADAPTIVE_THRESH)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (grid_w,grid_h), corners2, ret)
            if verbose:
                st.image(img)
        else:
            st.text(name + ' failed!')

    # 2. CALIBRATE: 
    # Distortion coeffs
    # dist = (k1, k2, p1, p2, k3)
    # Intrinsic params: 
    # focal length (f1, f2), 
    # optical center (c1, c2)
    # mtx = ((f1, 0, c1), (0, f2, c2), (0, 0, 1))
    # Extrinsic params
    # rvecs = rodrigues rotation vectors
    # tvecs = translation vectors
    retval, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, calibration_shape, None, None)
    print('camera matrix\n', mtx)
    print(f'returned error  {retval:.2f} pixels')
    if verbose:
        st.text(f'error  {retval:.2f} pixels')
    return mtx, dist

def reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
        print('err', error)
    mean_error /= len(objpoints)
    print(f"total error: {mean_error}")
    return mean_error

def undistort(img, mtx, dist):
    # refine camera matrix
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # do not refine
    # dst = cv.undistort(img, mtx, dist, None, None)
    return dst

if __name__ == '__main__':
    import os
    from PIL import Image
    images = {}
    path = '../data/example2/good/'
    for fp in os.listdir(path):
        with open(path + fp, 'rb') as f:
            image = Image.open(f)
            img_array = np.array(image)
            images[fp] = img_array
    params = calibrate(images, grid_w=49, grid_h=24, verbose=False)
    undistorted = undistort(img_array, *params)