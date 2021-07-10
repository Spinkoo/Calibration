import numpy as np
import cv2
import glob


def get_images_from_file(path):
    images = glob.glob(path+'*.png')
    imgs = []
    for fname in images :
        frame = cv2.imread(fname)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgs.append(gray)
    return imgs

#Press the button s in order to use the current image in the calibration process
def get_images_from_video(path):
    imgs = []
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.isOpened():
            
            ret, frame = cap.read()
            if frame is None : 
                break
            cv2.imshow("",frame)

            if cv2.waitKey(30) & 0xFF == ord('s'):
                n+=1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                imgs.append(gray)
    return imgs
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#define chess board dim
chess_h = 9
chess_w = 6
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((chess_w*chess_h,3), np.float32)
objp[:,:2] = np.mgrid[0:chess_h,0:chess_w].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = []
imgpoints = [] 
n = 0

images = get_images_from_video('./calib.mp4')

#or from a file that contains the sequence of images
#get_images_from_file('./')

for gray in images:  
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chess_h,chess_w), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        print('Chess board treated')
        n+=1

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret,mtx,dist,rvecs,tvecs)
