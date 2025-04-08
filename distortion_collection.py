import cv2
import numpy as np

# 카메라 행렬 (Calibration 결과 입력)
mtx = np.array([
    [673.42682005, 0, 245.84823196],
    [0, 670.85149225, 351.89660938],
    [0, 0, 1]
])

# 왜곡 계수
dist = np.array([0.14594799, -0.91663695, -0.00058594, 0.00784773, 2.09888954])

# 영상 불러오기
cap = cv2.VideoCapture('./data/chess.mp4')  # ← 영상 경로 확인!

if not cap.isOpened():
    raise FileNotFoundError(" 비디오 파일을 열 수 없습니다.")

# 영상 정보 읽기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 출력용 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('undistorted_output.avi', fourcc, fps, (width, height))

# 새 카메라 행렬 계산
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 왜곡 보정
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # 출력
    out.write(undistorted)

    # 보기용
    cv2.imshow('Original', frame)
    cv2.imshow('Undistorted', undistorted)
    if cv2.waitKey(10) & 0xFF == 27:  # ESC 누르면 종료
        break

cap.release()
out.release()
cv2.destroyAllWindows()
