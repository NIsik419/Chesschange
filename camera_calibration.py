import cv2
import numpy as np

# 체스보드 내부 코너 개수 (가로 x 세로)
chessboard_size = (9, 6)

# 월드 좌표계의 기준점 생성
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D 점
imgpoints = []  # 2D 점

# 영상 파일 열기
cap = cv2.VideoCapture('./data/chess.mp4')
frame_count = 0
frame_interval = 10  # 10프레임마다 추출

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret_corners:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_corners)
        cv2.imshow('Corners', frame)
        if cv2.waitKey(100) == 27:
            break

cap.release()
cv2.destroyAllWindows()

# 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(" Camera matrix:\n", mtx)
print(" Distortion Coefficients:\n", dist)

# RMSE 계산
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error

print(" RMSE:", total_error / len(objpoints))
