import numpy as np
import cv2 as cv
import glob

# 25mm per square
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calib_pics/*.jpg')

for image_file in images:
    img = cv.imread(image_file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    if ret:
        objpoints.append(objp)
        # find more accurate corners
        acc_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, 1), criteria)
        imgpoints.append(acc_corners)

        img = cv.drawChessboardCorners(img, (7, 6), acc_corners, ret)
        cv.imshow("calibration image", img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# test undistortion on one image

img = cv.imread("img.jpg")
height, width = img.shape[:2]
new_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

dst = cv.undistort(img, mtx, dist, None, new_matrix)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite("calibresult.jpg", dst)

mean_err = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    err = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_err += error

mean_err = mean_err / len(objpoints)
print(mean_err)

