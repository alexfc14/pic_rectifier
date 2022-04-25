import numpy as np
import cv2 as cv
import glob

import streamlit as st

def calibrate(images, grid_w=9, grid_h=6, verbose=True):
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
    for img in images:
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

    # 2. CALIBRATE
    retval, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, calibration_shape, None, None)
    return mtx, dist

def undistort(img, mtx, dist):
    # refine camera matrix
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst
